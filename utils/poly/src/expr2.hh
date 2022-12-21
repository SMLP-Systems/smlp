/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr.hh"
#include "algebraics.hh"

namespace smlp {

/* Definition of 'term2' terms and (quantifier-free) 'form2' formulas. */

struct term2;
struct form2;

/* A 'form2' formula is either:
 * - prop2: two term2 expressions compared using a cmp_t
 * - lbop2: logical binary connective of n form2 values: AND, OR
 * - lneg2: logical negation of a form2
 */

struct prop2 {
	cmp_t cmp; sptr<term2> left, right;
	bool operator==(const prop2 &b) const;
	std::strong_ordering operator<=>(const prop2 &b) const;
};
struct lbop2 {
	enum { AND, OR } op; vec<sptr<form2>> args;
	bool operator==(const lbop2 &b) const;
	std::strong_ordering operator<=>(const lbop2 &b) const;

	friend decltype(op) operator!(decltype(op) o)
	{
		switch (o) {
		case AND: return OR;
		case OR: return AND;
		}
		unreachable();
	}
};
struct lneg2 {
	sptr<form2> arg;
	bool operator==(const lneg2 &b) const;
	std::strong_ordering operator<=>(const lneg2 &b) const;
};

inline const char *lbop_s[] = { "and", "or" };

namespace detail {
str to_string(const sptr<form2> &, bool let);
str to_string(const sptr<term2> &, bool let);
}

enum class type : int { INT, REAL };

struct form2 : sumtype<prop2,lbop2,lneg2>
             , std::enable_shared_from_this<form2> {

	using sumtype<prop2,lbop2,lneg2>::sumtype;

	friend str to_string(const sptr<form2> &f, bool let = true)
	{
		return detail::to_string(f, let);
	}
};

/* An 'term2' expression is either:
 * - name : identifier, same as in 'expr'
 * - bop2 : binary arithmetic operation, same as 'bop' in 'expr'
 * - uop2 : unary arithmetic operation, same as 'uop' in 'expr'
 * - cnst2: disjoint union of integers (kay::Z), rationals (kay::Q) and strings
 * - ite2 : if-then-else: condition (form2) and two term2 expressions
 *
 * Note that in contrast to 'expr' expressions, there is no 'call' operator.
 * Another difference to 'expr' is 'cnst2': numeric literals have been
 * interpreted (and rationals potentially reduced).
 */

struct ite2 {
	sptr<form2> cond; sptr<term2> yes, no;
	bool operator==(const ite2 &b) const;
	std::strong_ordering operator<=>(const ite2 &b) const;
};
struct bop2 {
	enum { ADD, SUB, MUL, } op; sptr<term2> left, right;
	bool operator==(const bop2 &b) const;
	std::strong_ordering operator<=>(const bop2 &b) const;
};
struct uop2 {
	enum { UADD, USUB, } op; sptr<term2> operand;
	bool operator==(const uop2 &b) const;
	std::strong_ordering operator<=>(const uop2 &b) const;
};

struct cnst2 {
	struct ZQA : sumtype<kay::Z,kay::Q,A> {

		using sumtype<kay::Z,kay::Q,A>::sumtype;

		friend str to_string(const ZQA &v)
		{
			return v.match(
			[](const auto &x) { return kay::to_string(x); },
			[](const A &x) { return to_string(x); }
			);
		}

		friend kay::Q to_Q(const ZQA &v)
		{
			return v.match(
			[](const kay::Z &x) { return kay::Q(x); },
			[](const kay::Q &x) { return x; },
			[](const Ap &a) { return to_Q(a); }
			);
		}

		std::strong_ordering operator<=>(const ZQA &b) const = default;
	} value;
	bool operator==(const cnst2 &b) const;
	std::strong_ordering operator<=>(const cnst2 &b) const = default;
};

struct term2 : sumtype<name,bop2,uop2,cnst2,ite2>
             , std::enable_shared_from_this<term2> {

	using sumtype<name,bop2,uop2,cnst2,ite2>::sumtype;

	friend str to_string(const sptr<term2> &t, bool let = true)
	{
		return detail::to_string(t, let);
	}
};

template <typename... Ts>
static inline sptr<term2> make2t(Ts &&... ts)
{
	return std::make_shared<term2>(std::forward<Ts>(ts)...);
}

template <typename... Ts>
static inline sptr<form2> make2f(Ts &&... ts)
{
	return std::make_shared<form2>(std::forward<Ts>(ts)...);
}

/* Short forms for creating propositional formulas: conjunction,
 * disjunction and negation */
inline sptr<form2> conj(vec<sptr<form2>> l)
{
	return make2f(lbop2 { lbop2::AND, { move(l) } });
}

inline sptr<form2> disj(vec<sptr<form2>> l)
{
	return make2f(lbop2 { lbop2::OR, { move(l) } });
}

inline sptr<form2> neg(sptr<form2> f)
{
	return make2f(lneg2 { move(f) });
}

/* Constants for true, false, 0 and 1 */
inline const sptr<form2> true2  = conj({});
inline const sptr<form2> false2 = disj({});
inline const sptr<term2> zero   = make2t(cnst2 { kay::Z(0) });
inline const sptr<term2> one    = make2t(cnst2 { kay::Z(1) });

/* Evaluate known function symbols in 'funs' that occur as a 'call' application
 * in the expr 'e'. Results in a term2 term. */
typedef sumtype<sptr<term2>,sptr<form2>> expr2s;
typedef hmap<str,fun<expr2s(vec<expr2s>)>> unroll_funs_t;
expr2s unroll_add(vec<expr2s> a);
expr2s unroll_sub(vec<expr2s> a);
expr2s unroll_mul(vec<expr2s> a);
expr2s unroll_expz(vec<expr2s> a);
expr2s unroll_and(vec<expr2s> a);
expr2s unroll_or(vec<expr2s> a);
expr2s unroll_not(vec<expr2s> a);
expr2s unroll_div_cnst(vec<expr2s> a);

sptr<term2> unroll_cnst_None(const cnst &c);
sptr<term2> unroll_cnst_ZQ(const cnst &c);
sptr<term2> unroll_cnst_Q(const cnst &c);

expr2s unroll(const expr &e, const unroll_funs_t &funs,
              fun<sptr<term2>(const cnst &)> ip_cnst = unroll_cnst_Q);

namespace expr2_ops {
static inline sptr<term2> operator*(sptr<term2> a, sptr<term2> b)
{
	return make2t(bop2 { bop2::MUL, move(a), move(b) });
}
static inline sptr<term2> & operator*=(sptr<term2> &a, sptr<term2> b)
{
	a = move(a) * move(b);
	return a;
}
static inline sptr<term2> operator+(sptr<term2> a, sptr<term2> b)
{
	return make2t(bop2 { bop2::ADD, move(a), move(b) });
}
static inline sptr<term2> & operator+=(sptr<term2> &a, sptr<term2> b)
{
	a = move(a) + move(b);
	return a;
}
static inline void fma(sptr<term2> &a, sptr<term2> b, sptr<term2> c)
{
	a += move(b) * move(c);
}
static inline sptr<term2> operator-(sptr<term2> a, sptr<term2> b)
{
	return make2t(bop2 { bop2::SUB, move(a), move(b) });
}
static inline sptr<term2> operator-=(sptr<term2> &a, sptr<term2> b)
{
	a = move(a) - move(b);
	return a;
}
static inline sptr<term2> operator+(sptr<term2> a)
{
	return make2t(uop2 { uop2::UADD, move(a) });
}
static inline sptr<term2> operator-(sptr<term2> a)
{
	return make2t(uop2 { uop2::USUB, move(a) });
}
static inline sptr<term2> var(str n)
{
	return make2t(name { move(n) });
}
static inline sptr<term2> cQ(kay::Q q)
{
	return make2t(cnst2 { move(q) });
}
template <char... cs>
static inline sptr<term2> operator""_Q()
{
	char str[] = { cs..., '\0', };
	return cQ(kay::Q_from_str(str));
}
static inline sptr<term2> pow(sptr<term2> a, kay::Z n)
{
	return *unroll_expz({ a, make2t(cnst2 { move(n) }) }).get<sptr<term2>>();
}

template <cmp_t c>
static inline sptr<form2> cmp(sptr<term2> a, sptr<term2> b)
{
	return make2f(prop2 { c, move(a), move(b) });
}

static inline sptr<form2> cmp(sptr<term2> a, cmp_t c, sptr<term2> b)
{
	return make2f(prop2 { c, move(a), move(b) });
}
}

static inline sptr<term2> ite(sptr<form2> c, sptr<term2> y, sptr<term2> n)
{
	return make2t(ite2 { move(c), move(y), move(n) });
}

/* Absolute value on term2, encoded by ite2 on comparing with zero */
inline sptr<term2> abs(const sptr<term2> &e)
{
	using namespace expr2_ops;
	return ite(cmp<LT>(e, zero), -e, e);
}

/* Substitute all 'name' expressions with id in 'repl' by another expression. */
sptr<term2> subst(const sptr<term2> &e, const hmap<str,sptr<term2>> &repl);
sptr<form2> subst(const sptr<form2> &f, const hmap<str,sptr<term2>> &repl);

/* Ground terms and ground formulas contain no variables (i.e., 'name') */
bool is_ground(const sptr<term2> &e);
bool is_ground(const sptr<form2> &f);

/* Constant folding, under the (potentially empty) assignment 'repl'.
 * Arithmetic (bop2, uop2, prop2) terms are only evaluated when all parameters
 * are constants. Logical connectives (lbop2) are evaluated in a short-circuit
 * fashion from beginning to end. */
sptr<term2> cnst_fold(const sptr<term2> &e, const hmap<str,sptr<term2>> &repl);
sptr<form2> cnst_fold(const sptr<form2> &f, const hmap<str,sptr<term2>> &repl);

/* Determine whether a formula or term contains non-linear (sub-)terms */
bool is_nonlinear(const sptr<form2> &f);
bool is_nonlinear(const sptr<term2> &e);

static inline bool is_linear(const sptr<term2> &e)
{
	return !is_ground(e) && !is_nonlinear(e);
}

/* Names of the free variables in a term2 or a form2. As there are no
 * quantifiers, all occurring variables are considered free. */
hset<str> free_vars(const sptr<term2> &f);
hset<str> free_vars(const sptr<form2> &f);

/* Returns NULL in case t contains ite2 { c, y, n } with var in free_vars(c).
 * Otherwise, if var is not free in c and the derivatives y' and n' of y and n
 * exist, the derivative of ite2 { c, y, n } is ite2 { c, y', n' }. */
sptr<term2> derivative(const sptr<term2> &t, const str &var);

/* Similar to cnst_fold(t, {}), but short-circuit evaluation is also performed
 * on arithmetic (bop2, uop2) terms. */
sptr<term2> simplify(const sptr<term2> &t);
sptr<form2> simplify(const sptr<form2> &t);

/* The negation normal form of a formula has all negations pushed into the atoms
 * (prop2). The result is free of lneg2 on the top level. The formulas making up
 * the conditions of ite2 terms are not touched. */
sptr<form2> to_nnf(const sptr<form2> &f);

sptr<form2> all_eq(opt<kay::Q> delta, const hmap<str,sptr<term2>> &m);

}
