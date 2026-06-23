//! A tiny SQL-ish front end — parse `SELECT ... FROM ... WHERE ... GROUP BY ...` into a [`Query`].
//!
//! This is deliberately a **small, hand-written** parser, not a full SQL engine: it covers exactly
//! the aggregate-filter-group surface the engine executes, so every query a user can type maps
//! one-to-one onto a runnable plan. Supported grammar (case-insensitive keywords):
//!
//! ```text
//! SELECT <agg> [, <agg>]* FROM <dataset> [WHERE <pred>] [GROUP BY <col> [, <col>]*]
//! <agg>   := COUNT(*) | SUM(col) | MIN(col) | MAX(col) | AVG(col)
//! <pred>  := <term> [ (AND|OR) <term> ]*        (left-assoc; AND/OR same precedence here)
//! <term>  := [NOT] col <op> <literal>
//! <op>    := = | != | <> | < | <= | > | >=
//! <literal> := number | 'single-quoted string' | "double-quoted string" | true | false | null
//! ```
//!
//! It does **not** support joins, subqueries, projections of raw columns, or `HAVING`/`ORDER BY` —
//! those are out of scope for the map-reduce core. Any unsupported syntax is a clear error (never a
//! panic), so the CLI can surface it. The parser is pure and fully unit-tested.

use crate::query::{Agg, CmpOp, Predicate, Query};
use anyhow::{Result, anyhow, bail};

/// Parse a SQL-ish query string into a [`Query`]. Errors carry a human-readable reason.
pub fn parse(input: &str) -> Result<Query> {
    let tokens = tokenize(input)?;
    let mut p = Parser { tokens, pos: 0 };
    let q = p.parse_query()?;
    if !p.at_end() {
        bail!("unexpected trailing input near token {}", p.peek().unwrap_or(&"<eof>".to_string()));
    }
    q.validate()?;
    Ok(q)
}

/// A token: a keyword/identifier/operator word, or a quoted string literal (kept distinct so a
/// quoted value is never mistaken for a keyword).
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Str(String),
}

/// Split the input into tokens. Operators (`<=`, `>=`, `!=`, `<>`, `=`, `<`, `>`) and the comma and
/// parens are their own tokens; quoted strings are captured whole. Whitespace separates words.
fn tokenize(input: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // Quoted string literal.
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != quote {
                i += 1;
            }
            if i >= chars.len() {
                bail!("unterminated string literal");
            }
            let s: String = chars[start..i].iter().collect();
            toks.push(Tok::Str(s));
            i += 1; // skip closing quote
            continue;
        }
        // Multi-char and single-char operators / punctuation.
        let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        if matches!(two.as_str(), "<=" | ">=" | "!=" | "<>") {
            toks.push(Tok::Word(two));
            i += 2;
            continue;
        }
        if matches!(c, '=' | '<' | '>' | ',' | '(' | ')' | '*') {
            toks.push(Tok::Word(c.to_string()));
            i += 1;
            continue;
        }
        // Bare word: identifier / keyword / number, terminated by whitespace, operator, or punct.
        let start = i;
        while i < chars.len() {
            let d = chars[i];
            if d.is_whitespace() || matches!(d, '=' | '<' | '>' | ',' | '(' | ')' | '*' | '\'' | '"')
            {
                break;
            }
            i += 1;
        }
        if i > start {
            toks.push(Tok::Word(chars[start..i].iter().collect()));
        }
    }
    Ok(toks)
}

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Peek the current token as a word (None at end or if it is a string literal).
    fn peek(&self) -> Option<&String> {
        match self.tokens.get(self.pos) {
            Some(Tok::Word(w)) => Some(w),
            _ => None,
        }
    }

    fn peek_tok(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    /// Consume and return the current token.
    fn next_tok(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// Consume a word that equals `kw` (case-insensitive); error otherwise.
    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        match self.next_tok() {
            Some(Tok::Word(w)) if w.eq_ignore_ascii_case(kw) => Ok(()),
            other => bail!("expected `{kw}`, found {:?}", other),
        }
    }

    /// Is the next token the keyword `kw` (case-insensitive)?
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(w) if w.eq_ignore_ascii_case(kw))
    }

    fn parse_query(&mut self) -> Result<Query> {
        self.expect_kw("SELECT")?;
        let aggregates = self.parse_agg_list()?;
        self.expect_kw("FROM")?;
        let dataset = self
            .next_word()
            .ok_or_else(|| anyhow!("expected dataset name after FROM"))?;

        let predicate = if self.is_kw("WHERE") {
            self.expect_kw("WHERE")?;
            self.parse_predicate()?
        } else {
            Predicate::True
        };

        let mut group_by = Vec::new();
        if self.is_kw("GROUP") {
            self.expect_kw("GROUP")?;
            self.expect_kw("BY")?;
            loop {
                let col = self.next_word().ok_or_else(|| anyhow!("expected column after GROUP BY"))?;
                group_by.push(col);
                if self.peek().map(|w| w == ",").unwrap_or(false) {
                    self.pos += 1; // consume comma
                } else {
                    break;
                }
            }
        }

        Ok(Query { dataset, aggregates, predicate, group_by })
    }

    /// Consume a bare word (identifier/number), not an operator/string. None otherwise.
    fn next_word(&mut self) -> Option<String> {
        match self.peek_tok() {
            Some(Tok::Word(w)) if !is_operator(w) => {
                let w = w.clone();
                self.pos += 1;
                Some(w)
            }
            _ => None,
        }
    }

    fn parse_agg_list(&mut self) -> Result<Vec<Agg>> {
        let mut aggs = Vec::new();
        loop {
            aggs.push(self.parse_agg()?);
            if self.peek().map(|w| w == ",").unwrap_or(false) {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(aggs)
    }

    fn parse_agg(&mut self) -> Result<Agg> {
        let func = self
            .next_word()
            .ok_or_else(|| anyhow!("expected an aggregate function in SELECT"))?;
        self.expect_word("(")?;
        let arg = match self.next_tok() {
            Some(Tok::Word(w)) => w,
            other => bail!("expected aggregate argument, found {:?}", other),
        };
        self.expect_word(")")?;
        let agg = match func.to_ascii_uppercase().as_str() {
            "COUNT" => Agg::Count, // arg ignored (`*` conventional)
            "SUM" => Agg::Sum(field_arg(&arg)?),
            "MIN" => Agg::Min(field_arg(&arg)?),
            "MAX" => Agg::Max(field_arg(&arg)?),
            "AVG" => Agg::Avg(field_arg(&arg)?),
            other => bail!("unknown aggregate function `{other}`"),
        };
        Ok(agg)
    }

    fn expect_word(&mut self, w: &str) -> Result<()> {
        match self.next_tok() {
            Some(Tok::Word(x)) if x == w => Ok(()),
            other => bail!("expected `{w}`, found {:?}", other),
        }
    }

    /// Parse a predicate: a chain of terms joined by AND/OR, left-associative.
    fn parse_predicate(&mut self) -> Result<Predicate> {
        let mut left = self.parse_term()?;
        loop {
            if self.is_kw("AND") {
                self.expect_kw("AND")?;
                let right = self.parse_term()?;
                left = Predicate::And(Box::new(left), Box::new(right));
            } else if self.is_kw("OR") {
                self.expect_kw("OR")?;
                let right = self.parse_term()?;
                left = Predicate::Or(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    /// Parse one comparison term, optionally negated: `[NOT] col <op> <literal>`.
    fn parse_term(&mut self) -> Result<Predicate> {
        let negate = if self.is_kw("NOT") {
            self.expect_kw("NOT")?;
            true
        } else {
            false
        };
        let field = self
            .next_word()
            .ok_or_else(|| anyhow!("expected a column name in WHERE term"))?;
        let op = self.parse_op()?;
        let value = self.parse_literal()?;
        let cmp = Predicate::Cmp { field, op, value };
        Ok(if negate { Predicate::Not(Box::new(cmp)) } else { cmp })
    }

    fn parse_op(&mut self) -> Result<CmpOp> {
        let w = match self.next_tok() {
            Some(Tok::Word(w)) => w,
            other => bail!("expected a comparison operator, found {:?}", other),
        };
        Ok(match w.as_str() {
            "=" => CmpOp::Eq,
            "!=" | "<>" => CmpOp::Ne,
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            other => bail!("unknown comparison operator `{other}`"),
        })
    }

    /// Parse a literal value: quoted string, number, boolean, or null.
    fn parse_literal(&mut self) -> Result<serde_json::Value> {
        match self.next_tok() {
            Some(Tok::Str(s)) => Ok(serde_json::Value::String(s)),
            Some(Tok::Word(w)) => Ok(literal_from_word(&w)),
            None => bail!("expected a literal value"),
        }
    }
}

/// A `COUNT(*)` argument is `*`; for a field aggregate the argument must be a real column name.
fn field_arg(arg: &str) -> Result<String> {
    if arg == "*" {
        bail!("`*` is only valid as COUNT(*) — field aggregates need a column name");
    }
    Ok(arg.to_string())
}

/// Convert a bare word literal to JSON: `true`/`false`/`null`, else a number, else a bare string.
fn literal_from_word(w: &str) -> serde_json::Value {
    match w.to_ascii_lowercase().as_str() {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" => return serde_json::Value::Null,
        _ => {}
    }
    if let Ok(n) = w.parse::<i64>() {
        return serde_json::json!(n);
    }
    if let Ok(f) = w.parse::<f64>() {
        return serde_json::json!(f);
    }
    serde_json::Value::String(w.to_string())
}

fn is_operator(w: &str) -> bool {
    matches!(w, "=" | "!=" | "<>" | "<" | "<=" | ">" | ">=" | "," | "(" | ")")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_simple_count() {
        let q = parse("SELECT COUNT(*) FROM events").unwrap();
        assert_eq!(q.dataset, "events");
        assert_eq!(q.aggregates, vec![Agg::Count]);
        assert_eq!(q.predicate, Predicate::True);
        assert!(q.group_by.is_empty());
    }

    #[test]
    fn parse_sum_with_where_and_group() {
        let q = parse("SELECT SUM(amount), COUNT(*) FROM sales WHERE region = 'EU' GROUP BY product")
            .unwrap();
        assert_eq!(q.dataset, "sales");
        assert_eq!(q.aggregates, vec![Agg::Sum("amount".into()), Agg::Count]);
        assert_eq!(q.group_by, vec!["product".to_string()]);
        match &q.predicate {
            Predicate::Cmp { field, op, value } => {
                assert_eq!(field, "region");
                assert_eq!(*op, CmpOp::Eq);
                assert_eq!(*value, json!("EU"));
            }
            other => panic!("unexpected predicate {other:?}"),
        }
    }

    #[test]
    fn parse_and_or_not() {
        let q = parse("SELECT AVG(v) FROM t WHERE a > 10 AND NOT b = 'x' OR c <= 3").unwrap();
        // Left-assoc: ((a>10 AND NOT b='x') OR c<=3)
        match &q.predicate {
            Predicate::Or(l, r) => {
                assert!(matches!(**l, Predicate::And(_, _)));
                assert!(matches!(**r, Predicate::Cmp { .. }));
            }
            other => panic!("unexpected predicate {other:?}"),
        }
    }

    #[test]
    fn case_insensitive_keywords() {
        let q = parse("select count(*) from t where x >= 1 group by y").unwrap();
        assert_eq!(q.aggregates, vec![Agg::Count]);
        assert_eq!(q.group_by, vec!["y".to_string()]);
    }

    #[test]
    fn operators_all_parse() {
        for (s, want) in [
            ("=", CmpOp::Eq),
            ("!=", CmpOp::Ne),
            ("<>", CmpOp::Ne),
            ("<", CmpOp::Lt),
            ("<=", CmpOp::Le),
            (">", CmpOp::Gt),
            (">=", CmpOp::Ge),
        ] {
            let q = parse(&format!("SELECT COUNT(*) FROM t WHERE a {s} 1")).unwrap();
            match q.predicate {
                Predicate::Cmp { op, .. } => assert_eq!(op, want, "op {s}"),
                _ => panic!("expected cmp"),
            }
        }
    }

    #[test]
    fn numeric_and_bool_and_null_literals() {
        let q = parse("SELECT COUNT(*) FROM t WHERE a = 3.5").unwrap();
        match q.predicate {
            Predicate::Cmp { value, .. } => assert_eq!(value, json!(3.5)),
            _ => panic!(),
        }
        let q = parse("SELECT COUNT(*) FROM t WHERE flag = true").unwrap();
        match q.predicate {
            Predicate::Cmp { value, .. } => assert_eq!(value, json!(true)),
            _ => panic!(),
        }
        let q = parse("SELECT COUNT(*) FROM t WHERE x != null").unwrap();
        match q.predicate {
            Predicate::Cmp { value, op, .. } => {
                assert_eq!(value, serde_json::Value::Null);
                assert_eq!(op, CmpOp::Ne);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn errors_are_graceful_not_panics() {
        assert!(parse("").is_err());
        assert!(parse("SELECT FROM t").is_err());
        assert!(parse("SELECT FOO(x) FROM t").is_err());
        assert!(parse("SELECT COUNT(*)").is_err()); // no FROM
        assert!(parse("SELECT SUM(*) FROM t").is_err()); // * only valid for COUNT
        assert!(parse("SELECT COUNT(*) FROM t WHERE a 1").is_err()); // missing op
        assert!(parse("SELECT COUNT(*) FROM t junk").is_err()); // trailing
        assert!(parse("SELECT COUNT(*) FROM t WHERE a = 'unterminated").is_err());
    }

    #[test]
    fn group_by_multiple_columns() {
        let q = parse("SELECT COUNT(*) FROM t GROUP BY a, b, c").unwrap();
        assert_eq!(q.group_by, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn double_quoted_string_literal() {
        let q = parse("SELECT COUNT(*) FROM t WHERE name = \"alice\"").unwrap();
        match q.predicate {
            Predicate::Cmp { value, .. } => assert_eq!(value, json!("alice")),
            _ => panic!(),
        }
    }
}
