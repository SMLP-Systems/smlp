
#include <nn.hh>

#define SAFE_UNROLL_MAX 0
#undef unreachable
#include "nn/nn-common.hh"

using namespace smlp;
using namespace iv::functions;
using namespace expr2_ops;

using scaler = affine1<double,double>;
using pt_scaler = pointwise<scaler>;

static sptr<term2>
apply_scaler(const scaler &sc, const sptr<term2> &in, bool clamp_outputs)
{
	sptr<term2> c = cQ(sc.a) * in + cQ(sc.b);
	if (clamp_outputs)
		c = ite(cmp<LT>(c, zero), zero, ite(cmp<GT>(c, one), one, c));
	return c;
}

static vec<sptr<term2>>
apply_scaler(const pt_scaler &sc,
             const vec<sptr<term2>> &in, bool clamp_outputs)
{
	size_t n = size(in);
	assert(n == size(sc.f));
	vec<sptr<term2>> in_scaled;
	in_scaled.reserve(n);
	for (size_t i=0; i<n; i++)
		in_scaled.emplace_back(apply_scaler(sc.f[i], in[i], clamp_outputs));
	return in_scaled;
}

pre_problem smlp::parse_nn(const char *gen_path, const char *hdf5_path,
                           const char *spec_path, const char *io_bounds,
                           const char *obj_bounds, bool clamp_inputs,
                           bool single_obj)
{
	kjson::json gen = iv::nn::common::json_parse(gen_path);
	bool no_grid = true;
	iv::nn::common::model_fun2 mf2(gen, hdf5_path, spec_path, io_bounds,
	                               no_grid);

	kjson::json io_bnds = iv::nn::common::json_parse(io_bounds);
	vec<sptr<term2>> in_vars;
	hmap<str,ival> in_bnds;
	vec<sptr<form2>> eta; /* conjunction, for candidates */

	domain dom;
	hmap<str,size_t> name2spec;
	for (ssize_t i=0; i<input_dim(mf2.spec); i++) {
		kjson::json s = mf2.spec.spec[mf2.spec.dom2spec[i]];
		str id = s["label"].get<str>();
		name2spec[id] = mf2.spec.dom2spec[i];

		kjson::json bnds = io_bnds[id];
		kay::Q lo = bnds["min"].get<kay::Q>();
		kay::Q hi = bnds["max"].get<kay::Q>();
		component c;
		if (s["range"] == "int")
			c.type = type::INT;
		else {
			assert(s["range"] == "float");
			c.type = type::REAL;
		}
		dom.emplace_back(id, move(c));
		in_vars.emplace_back(var(id));
		in_bnds.emplace(move(id), ival { lo, hi });
		if (s.contains("safe")) {
			vec<sptr<form2>> safe;
			for (const kjson::json &v : s["safe"])
				safe.emplace_back(cmp<EQ>(in_vars.back(), cQ(v.get<kay::Q>())));
			eta.emplace_back(disj(move(safe)));
		}
	}
	// dump_smt2(stdout, dom);
	// dump_smt2(stdout, lbop2 { lbop2::AND, move(in_bnds) });

	const opt_fun<pt_scaler> &in_scaler_opt = mf2.in_scaler;
	assert(in_scaler_opt);
	vec<sptr<term2>> in_scaled = apply_scaler(*in_scaler_opt, in_vars, clamp_inputs);

	/* sequential_dense is
	 *   finite_composition<dense_layer<keras::activation>>
	 *
	 * dense_layer<..> is
	 *   composition<affine_matrix<float>,keras::activation>
	 *
	 * keras::activation is
	 *   var_fun<keras::linear,keras::relu>
	 */
	iv::nn::common::sequential_dense &m = mf2.model;
	// size_t n_layers = size(m.funs);
	size_t layer = 0;
	vec<sptr<term2>> out = in_scaled;
	for (const auto &f : m.funs) {
		const iv::nn::common::affine_matrix<float> &am = std::get<0>(f.t);
		const auto &kernel = am.a; /* matrix<float> */
		const vec<float> &bias = am.b;
		note(mod_nn,"layer %zu: w: %zu, h: %zu, bias: %zu\n",
		     layer, width(kernel), height(kernel), size(bias));
		/* matrix-vector product */
		assert(width(kernel) == size(out));
		assert(height(kernel) == size(bias));
		vec<sptr<term2>> next;
		for (float b : bias)
			next.push_back(cQ(b));
		for (size_t y=0; y<height(kernel); y++)
			for (size_t x=0; x<width(kernel); x++)
				fma(next[y], cQ(kernel(x,y)), out[x]);
		match(std::get<1>(f.t).f.f,
		[&](const iv::nn::common::keras::relu &) {
			for (sptr<term2> &e : next)
				e = ite(cmp<LT>(e, zero), zero, e);
		},
		[](const iv::nn::common::keras::linear &) {});
		layer++;
		out = move(next);
	}

	assert(compatible((dim)size(out), output_dim(mf2.spec)));
	if (mf2.out_scaler)
		out = apply_scaler(*mf2.out_scaler, out, false);

	if (mf2.objective.type != smlp_response::SMLP_RESPONSE_ID) {
		std::stringstream ss;
		ss << mf2.objective;
		MDIE(mod_nn,1,"unsupported objective function type, only "
		              "identity is: %s\n", ss.str().c_str());
	}

	vec<str> out_names;
	for (kjson::json comp : mf2.spec.spec)
		if (comp["type"] == "response")
			out_names.emplace_back(comp["label"].get<std::string_view>());

	assert(size(out_names) == size(out));

	hmap<str,sptr<term2>> outs;
	for (size_t i=0; i<size(out); i++)
		outs[out_names[i]] = out[i];


	hmap<str,ival> func_bounds;
	if (obj_bounds) {
		/* basically rebuild mf2.objective_scaler() for all outputs and
		 * with kay::Q instead of double */
		auto sc = [&](auto bnds) {
			for (size_t i : mf2.spec.resp2spec) {
				str lbl = mf2.spec.spec[i]["label"].get<str>();
				const auto &b = bnds[lbl];
				ival v = {
					b["min"].template get<kay::Q>(),
					b["max"].template get<kay::Q>(),
				};
				auto [it,ins] = func_bounds.emplace(move(lbl), move(v));
				assert(ins);
			}
		};
		using namespace iv::nn::common;
		try {
			sc(model_fun2::Table_proxy(file(obj_bounds, "r"), true));
		} catch (const iv::table_exception &ex) {
			sc(json_parse(obj_bounds));
		}
	}

	sptr<term2> obj;
	if (single_obj) {
		/* apply mf2.objective: select right output(s) */
		str obj_name = gen["objective"].get<std::string>();
		note(mod_nn,"obj '%s' response idx: %zd\n",
		     obj_name.c_str(), mf2.objective.idx[0]);
		assert(mf2.objective.type == smlp_response::SMLP_RESPONSE_ID);
		ssize_t idx = mf2.objective.idx[0];
		assert(idx >= 0);
		assert((size_t)idx < size(out));
		// obj = out[idx];
		assert(obj_name == out_names[idx]);
		obj = var(move(obj_name));
	} else {
		/* unknown objective(s) and we are not instructed to take the
		 * information from GEN. */
	}

	/* construct theta */
	auto theta = [spec=move(mf2.spec.spec), name2spec=move(name2spec)]
	             (opt<kay::Q> delta, const hmap<str,sptr<term2>> &v) {
		using expr2_ops::operator-;
		vec<sptr<form2>> conj;
		for (const auto &[n,e] : v) {
			auto it = name2spec.find(n);
			assert(it != name2spec.end());
			kjson::json sp = spec[it->second];
			sptr<term2> nm = var(n);
			sptr<term2> r;
			if (sp.contains("rad-abs")) {
				kay::Q rad = sp["rad-abs"].get<kay::Q>();
				if (delta)
					rad *= (1 + *delta);
				r = cQ(move(rad));
			} else if (sp.contains("rad-rel")) {
				kay::Q rad = sp["rad-rel"].get<kay::Q>();
				if (delta)
					rad *= (1 + *delta);
				r = cQ(move(rad));
				r = move(r) * abs(!delta ? e : nm);
			} else if (sp["type"] == "input")
				continue;
			else
				MDIE(mod_nn,1,".spec contains neither 'rad-abs' "
				              "nor 'rad-rel' for '%s'\n",
				     n.c_str());
			conj.emplace_back(cmp<LE>(abs(nm - e), move(r)));
		}
		return smlp::conj(move(conj));
	};

	return pre_problem {
		move(dom),
		move(obj),
		move(outs),
		move(func_bounds),
		move(in_bnds),
		conj(move(eta)),
		true2,
		move(theta),
	};
}
