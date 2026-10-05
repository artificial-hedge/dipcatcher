# Hand-checked FX-1 cases. Expectation dicts match Output.model_dump(mode="json").
# No fx1 import.
#
# Every value below was derived before the operation was run, from closed-form
# math or from an independent oracle (own NumPy SVD, brute-force enumeration,
# textbook permutation/subset averages, scipy.interpolate). "approx" is False
# only where every expected number is a dyadic rational that the operation's own
# exact rational path must reproduce bit-for-bit; it is True where an irrational
# (a singular value, log 2, 1/6) enters the comparison.

CASES = {
    "features.singular_spectrum_analysis": {
        "arguments": {
            "values": [1.0, 3.0, 2.0, 4.0, 3.0],
            "window": 2,
            "groups": [[0]],
        },
        "expect": {
            "observation_count": 5,
            "window": 2,
            "trajectory_columns": 4,
            "component_count": 2,
            "normalization_power_of_two": 3,
            "normalized_singular_values": [0.9943163664093607, 0.27172589771769273],
            "numerical_rank": 2,
            "unselected_component_indices": [1],
            "all_components_selected": False,
            "all_component_reconstruction": [1.0, 3.0, 2.0, 4.0, 3.0],
            "all_component_maximum_absolute_error": 0.0,
            "all_component_reconstruction_root_mean_square_error": {"value": 0.0},
            "svd_relative_frobenius_reconstruction_error": {"value": 0.0},
            "anti_diagonal_counts": [1, 2, 2, 2, 1],
            "close_adjacent_singular_index_pairs": [],
            "close_pairs_split_between_selection_groups": [],
            "status": "reconstructed",
            "centering": "none",
            "rank_threshold_filters_components": False,
            "groups": [
                {
                    "component_indices": [0],
                    "below_rank_threshold_indices": [],
                    "singular_squared_mass_fraction": 0.9305082696560132,
                }
            ],
            "residuals": [
                -0.9176129952966962,
                0.7570341530673937,
                -0.7335869761760809,
                0.7617235884456565,
                -0.6861983541924963,
            ],
            "maximum_absolute_residual": 0.9176129952966962,
        },
        "approx": True,
        "note": (
            "H=[[1,3,2,4],[3,2,4,3]] gives H H^T=[[30,29],[29,38]], trace 68 and det 299, so the "
            "raw sigma^2 are 34+-sqrt(857). max|x|=4.0 has frexp exponent 3, so the embedded "
            "trajectory is divided by 2^3=8 before the SVD -- losslessly, since the input becomes "
            "[1/8,3/8,1/4,1/2,3/8] -- and normalized_singular_values = sqrt(34+-sqrt 857)/8. The "
            "mass fraction is scale-invariant at (34+sqrt 857)/68. Anti-diagonal counts for L=2, "
            "K=4 are [1,2,2,2,1]. Rank 2 equals component_count, so summing both hankelized "
            "rank-one terms must return the input; the residual is x minus component 0 alone, "
            "reproduced by an own NumPy-SVD oracle (sigma0/sigma1=3.66, so the split is unique). "
            "The 0.7226 gap dwarfs the 9.94e-13 rank threshold, so no adjacent pair is close."
        ),
    },
    "features.linear_constraint_feasibility": {
        "arguments": {
            "variables": ["x"],
            "constraints": [
                {"constraint_id": "upper", "coefficients": [2.0], "upper_bound": 10.0},
                {"constraint_id": "lower", "coefficients": [-3.0], "upper_bound": -21.0},
            ],
        },
        "expect": {
            "status": "infeasible",
            "variables": ["x"],
            "original_constraint_count": 2,
            "exact_witness": None,
            "original_constraint_slacks": None,
            "rounded_witness_satisfies_original": None,
            "farkas_certificate": {
                "source_multipliers": [
                    {
                        "constraint_row_index": 0,
                        "constraint_id": "upper",
                        "multiplier": {"numerator": "1", "denominator": "4", "value": 0.25},
                    },
                    {
                        "constraint_row_index": 1,
                        "constraint_id": "lower",
                        "multiplier": {
                            "numerator": "1",
                            "denominator": "6",
                            "value": 0.16666666666666666,
                        },
                    },
                ],
                "combined_coefficients": [{"numerator": "0", "denominator": "1", "value": 0.0}],
                "combined_upper_bound": {"numerator": "-1", "denominator": "1", "value": -1.0},
                "nonnegative_multipliers_verified": True,
                "zero_left_side_verified": True,
                "negative_unit_right_side_verified": True,
            },
            "elimination_stages": [
                {
                    "variable_index": 0,
                    "input_rows": 2,
                    "positive_rows": 1,
                    "negative_rows": 1,
                    "zero_rows": 0,
                    "candidate_pair_count": 1,
                    "generated_rows": 1,
                    "retained_rows": 0,
                    "contradiction_found": True,
                }
            ],
            "total_generated_rows": 3,
            "maximum_retained_rows": 2,
            "exact_certificate_verified": True,
            "integer_feasibility_verified": False,
        },
        "approx": True,
        "note": (
            "2x<=10 means x<=5 and -3x<=-21 means x>=7, so no real x exists. Scaling each row by "
            "its first coefficient gives x<=5 with weight 1/2 and -x<=-7 with weight 1/3; the one "
            "Fourier-Motzkin pair combines them into 0<=-2, normalized to 0<=-1 with multipliers "
            "(1/4, 1/6). Independent check of Farkas' alternative: lambda=(1/4,1/6)>=0, "
            "lambda A = 1/4*2 + 1/6*(-3) = 0 and lambda b = 1/4*10 + 1/6*(-21) = -1."
        ),
    },
    "features.maximum_flow": {
        "arguments": {
            "vertices": ["s", "a", "b", "t"],
            "edges": [
                {"edge_id": "sa", "source": "s", "target": "a", "capacity": 3.0},
                {"edge_id": "sb", "source": "s", "target": "b", "capacity": 2.0},
                {"edge_id": "at", "source": "a", "target": "t", "capacity": 2.0},
                {"edge_id": "bt", "source": "b", "target": "t", "capacity": 3.0},
                {"edge_id": "ab", "source": "a", "target": "b", "capacity": 1.0},
            ],
            "source": "s",
            "sink": "t",
        },
        "expect": {
            "vertices": ["s", "a", "b", "t"],
            "source_index": 0,
            "sink_index": 3,
            "maximum_flow": {"numerator": "5", "denominator": "1", "value": 5.0},
            "minimum_cut_capacity": {"numerator": "5", "denominator": "1", "value": 5.0},
            "source_side_vertex_indices": [0],
            "sink_side_vertex_indices": [1, 2, 3],
            "cut_edge_row_indices": [0, 1],
            "edge_flows": [
                {
                    "edge_row_index": 0,
                    "edge_id": "sa",
                    "source_index": 0,
                    "target_index": 1,
                    "capacity": 3.0,
                    "flow": {"numerator": "3", "denominator": "1", "value": 3.0},
                    "forward_residual_capacity": {
                        "numerator": "0",
                        "denominator": "1",
                        "value": 0.0,
                    },
                    "crosses_minimum_cut": True,
                },
                {
                    "edge_row_index": 1,
                    "edge_id": "sb",
                    "source_index": 0,
                    "target_index": 2,
                    "capacity": 2.0,
                    "flow": {"numerator": "2", "denominator": "1", "value": 2.0},
                    "forward_residual_capacity": {
                        "numerator": "0",
                        "denominator": "1",
                        "value": 0.0,
                    },
                    "crosses_minimum_cut": True,
                },
                {
                    "edge_row_index": 2,
                    "edge_id": "at",
                    "source_index": 1,
                    "target_index": 3,
                    "capacity": 2.0,
                    "flow": {"numerator": "2", "denominator": "1", "value": 2.0},
                    "forward_residual_capacity": {
                        "numerator": "0",
                        "denominator": "1",
                        "value": 0.0,
                    },
                    "crosses_minimum_cut": False,
                },
                {
                    "edge_row_index": 3,
                    "edge_id": "bt",
                    "source_index": 2,
                    "target_index": 3,
                    "capacity": 3.0,
                    "flow": {"numerator": "3", "denominator": "1", "value": 3.0},
                    "forward_residual_capacity": {
                        "numerator": "0",
                        "denominator": "1",
                        "value": 0.0,
                    },
                    "crosses_minimum_cut": False,
                },
                {
                    "edge_row_index": 4,
                    "edge_id": "ab",
                    "source_index": 1,
                    "target_index": 2,
                    "capacity": 1.0,
                    "flow": {"numerator": "1", "denominator": "1", "value": 1.0},
                    "forward_residual_capacity": {
                        "numerator": "0",
                        "denominator": "1",
                        "value": 0.0,
                    },
                    "crosses_minimum_cut": False,
                },
            ],
            "vertex_net_outflows": [
                {"numerator": "5", "denominator": "1", "value": 5.0},
                {"numerator": "0", "denominator": "1", "value": 0.0},
                {"numerator": "0", "denominator": "1", "value": 0.0},
                {"numerator": "-5", "denominator": "1", "value": -5.0},
            ],
            "exact_capacity_and_conservation_verified": True,
            "exact_flow_cut_equality_verified": True,
            "optimizer_uniqueness_verified": False,
        },
        "approx": False,
        "note": (
            "Brute force over all four s-t cuts gives capacities 5 ({s}), 5 ({s,a}), 5 ({s,a,b}) "
            "and 6 ({s,b}), so max-flow = min-cut = 5. The flow of value 5 is unique: t admits "
            "only 2+3=5, so at=2 and bt=3 are saturated; then conservation at b forces ab=1 and "
            "sb=2, and at a forces sa=3 (a scipy linprog max/min of each edge flow at value 5 "
            "returns the same number both ways). With both source edges saturated the residual "
            "reachability from s is {s} alone, so the certified cut is rows 0 and 1, capacity 5, "
            "and every forward residual is 0."
        ),
    },
    "features.minimum_spanning_forest": {
        "arguments": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                {"edge_id": "ab", "first_vertex": "a", "second_vertex": "b", "weight": 1.0},
                {"edge_id": "bc", "first_vertex": "b", "second_vertex": "c", "weight": 2.0},
                {"edge_id": "ac", "first_vertex": "a", "second_vertex": "c", "weight": 3.0},
                {"edge_id": "cd", "first_vertex": "c", "second_vertex": "d", "weight": 4.0},
                {"edge_id": "ad", "first_vertex": "a", "second_vertex": "d", "weight": 5.0},
            ],
        },
        "expect": {
            "vertices": ["a", "b", "c", "d"],
            "selected_edge_row_indices": [0, 1, 3],
            "total_weight": {"numerator": "7", "denominator": "1", "value": 7.0},
            "components": [
                {
                    "vertex_indices": [0, 1, 2, 3],
                    "selected_edge_row_indices": [0, 1, 3],
                    "total_weight": {"numerator": "7", "denominator": "1", "value": 7.0},
                    "unique_minimum_edge_set": True,
                }
            ],
            "edge_classifications": [
                {
                    "edge_row_index": 0,
                    "edge_id": "ab",
                    "first_vertex_index": 0,
                    "second_vertex_index": 1,
                    "weight": 1.0,
                    "selected": True,
                    "classification": "selected_required",
                    "maximum_selected_path_weight": None,
                },
                {
                    "edge_row_index": 1,
                    "edge_id": "bc",
                    "first_vertex_index": 1,
                    "second_vertex_index": 2,
                    "weight": 2.0,
                    "selected": True,
                    "classification": "selected_required",
                    "maximum_selected_path_weight": None,
                },
                {
                    "edge_row_index": 2,
                    "edge_id": "ac",
                    "first_vertex_index": 0,
                    "second_vertex_index": 2,
                    "weight": 3.0,
                    "selected": False,
                    "classification": "excluded_heavier_than_path",
                    "maximum_selected_path_weight": 2.0,
                },
                {
                    "edge_row_index": 3,
                    "edge_id": "cd",
                    "first_vertex_index": 2,
                    "second_vertex_index": 3,
                    "weight": 4.0,
                    "selected": True,
                    "classification": "selected_required",
                    "maximum_selected_path_weight": None,
                },
                {
                    "edge_row_index": 4,
                    "edge_id": "ad",
                    "first_vertex_index": 0,
                    "second_vertex_index": 3,
                    "weight": 5.0,
                    "selected": False,
                    "classification": "excluded_heavier_than_path",
                    "maximum_selected_path_weight": 4.0,
                },
            ],
            "unique_minimum_edge_set": True,
            "exchange_witnesses": [],
            "exchangeable_unselected_edge_count": 0,
            "omitted_exchange_witness_count": 0,
            "exact_cycle_optimality_verified": True,
            "external_graph_verified": False,
        },
        "approx": False,
        "note": (
            "Brute force over all 32 edge subsets finds exactly 8 spanning trees, the cheapest "
            "being {ab,bc,cd} at 1+2+4=7 (next best 8), so the optimum is unique; distinct "
            "weights already force that. The cycle property: ac=3 closes the path a-b-c whose "
            "heaviest edge is bc=2, and 3>2 excludes it; ad=5 closes a-b-c-d whose heaviest edge "
            "is cd=4, and 5>4 excludes it. No unselected edge ties its path maximum, so there are "
            "no exchange witnesses and every selected edge is required."
        ),
    },
    "features.multilinear_grid_interpolation": {
        "arguments": {
            "axes": [
                {"name": "x", "coordinates": [0.0, 2.0]},
                {"name": "y", "coordinates": [0.0, 4.0]},
            ],
            "values": [0.0, 4.0, 8.0, 12.0],
            "queries": [[1.0, 2.0]],
            "integral_boxes": [[[0.0, 2.0], [0.0, 4.0]]],
        },
        "expect": {
            "axis_names": ["x", "y"],
            "shape": [2, 2],
            "strides": [2, 1],
            "node_count": 4,
            "arithmetic": "bounded_exact_supplied_binary_floats",
            "flattening": "last_axis_fastest",
            "field_fitted": False,
            "queries": [
                {
                    "query_row": 0,
                    "evaluated_coordinates": [1.0, 2.0],
                    "clipped_axes": [],
                    "lower_cell_indices": [0, 0],
                    "corner_value_rows": [0, 1, 2, 3],
                    "corner_weights": [
                        {"value": 0.25, "underflow": False},
                        {"value": 0.25, "underflow": False},
                        {"value": 0.25, "underflow": False},
                        {"value": 0.25, "underflow": False},
                    ],
                    "interpolated_value": {"value": 6.0, "underflow": False},
                    "coordinate_gradient": [
                        {"value": 4.0, "underflow": False},
                        {"value": 1.0, "underflow": False},
                    ],
                }
            ],
            "integrals": [
                {
                    "box_row": 0,
                    "orientation": 1,
                    "contributing_node_count": 4,
                    "integral": {"value": 48.0, "underflow": False},
                }
            ],
            "availability_or_units_verified": False,
        },
        "approx": False,
        "note": (
            "Last-axis-fastest values [0,4,8,12] on x=[0,2], y=[0,4] are exactly f(x,y)=4x+y, "
            "which is multilinear, so interpolation must reproduce it: the cell midpoint (1,2) "
            "has t=u=1/2, four equal corner weights 1/4, value 4*1+2=6 and gradient (4,1) "
            "(scipy's RegularGridInterpolator agrees: 6.0). The box integral of 4x+y over "
            "[0,2]x[0,4] is int_0^2 (16x+8) dx = 32+16 = 48; separable hat integrals give x "
            "weights [1,1] and y weights [2,2], and 1*2*0+1*2*4+1*2*8+1*2*12 = 48 over 4 nodes."
        ),
    },
    "features.set_function_attribution": {
        "arguments": {
            "names": ["a", "b", "c"],
            "values": [0.0, 1.0, 2.0, 9.0, 3.0, 4.0, 5.0, 24.0],
        },
        "expect": {
            "player_count": 3,
            "coalition_count": 8,
            "baseline": {"numerator": "0", "denominator": "1", "value": 0.0},
            "grand_coalition_value": {"numerator": "24", "denominator": "1", "value": 24.0},
            "grand_coalition_gain": {"numerator": "24", "denominator": "1", "value": 24.0},
            "attributions": [
                {
                    "player_index": 0,
                    "name": "a",
                    "shapley": {"numerator": "8", "denominator": "1", "value": 8.0},
                    "banzhaf": {"numerator": "7", "denominator": "1", "value": 7.0},
                    "standalone_gain": {"numerator": "1", "denominator": "1", "value": 1.0},
                    "grand_coalition_removal_loss": {
                        "numerator": "19",
                        "denominator": "1",
                        "value": 19.0,
                    },
                    "null_player": False,
                    "negative_marginal_count": 0,
                },
                {
                    "player_index": 1,
                    "name": "b",
                    "shapley": {"numerator": "9", "denominator": "1", "value": 9.0},
                    "banzhaf": {"numerator": "8", "denominator": "1", "value": 8.0},
                    "standalone_gain": {"numerator": "2", "denominator": "1", "value": 2.0},
                    "grand_coalition_removal_loss": {
                        "numerator": "20",
                        "denominator": "1",
                        "value": 20.0,
                    },
                    "null_player": False,
                    "negative_marginal_count": 0,
                },
                {
                    "player_index": 2,
                    "name": "c",
                    "shapley": {"numerator": "7", "denominator": "1", "value": 7.0},
                    "banzhaf": {"numerator": "6", "denominator": "1", "value": 6.0},
                    "standalone_gain": {"numerator": "3", "denominator": "1", "value": 3.0},
                    "grand_coalition_removal_loss": {
                        "numerator": "15",
                        "denominator": "1",
                        "value": 15.0,
                    },
                    "null_player": False,
                    "negative_marginal_count": 0,
                },
            ],
            "pair_interactions": [
                {
                    "first_player": 0,
                    "second_player": 1,
                    "shapley_interaction": {"numerator": "12", "denominator": "1", "value": 12.0},
                },
                {
                    "first_player": 0,
                    "second_player": 2,
                    "shapley_interaction": {"numerator": "6", "denominator": "1", "value": 6.0},
                },
                {
                    "first_player": 1,
                    "second_player": 2,
                    "shapley_interaction": {"numerator": "6", "denominator": "1", "value": 6.0},
                },
            ],
            "nonzero_dividend_count_by_order": [0, 3, 1, 1],
            "dividend_sum_by_order": [
                {"numerator": "0", "denominator": "1", "value": 0.0},
                {"numerator": "6", "denominator": "1", "value": 6.0},
                {"numerator": "6", "denominator": "1", "value": 6.0},
                {"numerator": "12", "denominator": "1", "value": 12.0},
            ],
            "monotone": True,
            "negative_marginal_count": 0,
            "monotonicity_findings": [],
            "omitted_monotonicity_findings": 0,
            "coalitions": [
                {
                    "mask": 0,
                    "member_indices": [],
                    "value": 0.0,
                    "mobius_dividend": {"numerator": "0", "denominator": "1", "value": 0.0},
                },
                {
                    "mask": 1,
                    "member_indices": [0],
                    "value": 1.0,
                    "mobius_dividend": {"numerator": "1", "denominator": "1", "value": 1.0},
                },
                {
                    "mask": 2,
                    "member_indices": [1],
                    "value": 2.0,
                    "mobius_dividend": {"numerator": "2", "denominator": "1", "value": 2.0},
                },
                {
                    "mask": 3,
                    "member_indices": [0, 1],
                    "value": 9.0,
                    "mobius_dividend": {"numerator": "6", "denominator": "1", "value": 6.0},
                },
                {
                    "mask": 4,
                    "member_indices": [2],
                    "value": 3.0,
                    "mobius_dividend": {"numerator": "3", "denominator": "1", "value": 3.0},
                },
                {
                    "mask": 5,
                    "member_indices": [0, 2],
                    "value": 4.0,
                    "mobius_dividend": {"numerator": "0", "denominator": "1", "value": 0.0},
                },
                {
                    "mask": 6,
                    "member_indices": [1, 2],
                    "value": 5.0,
                    "mobius_dividend": {"numerator": "0", "denominator": "1", "value": 0.0},
                },
                {
                    "mask": 7,
                    "member_indices": [0, 1, 2],
                    "value": 24.0,
                    "mobius_dividend": {"numerator": "12", "denominator": "1", "value": 12.0},
                },
            ],
            "offset": 0,
            "has_more": False,
            "exact_shapley_efficiency_checked": True,
            "model_or_causal_importance_verified": False,
        },
        "approx": False,
        "note": (
            "The table is v = 1*u_a + 2*u_b + 3*u_c + 6*u_ab + 12*u_abc in unanimity games, so "
            "its Mobius dividends are exactly those coefficients: d(a)=1, d(b)=2, d(c)=3, "
            "d(ab)=6, d(abc)=12 and d(ac)=d(bc)=0 (also from the signed-subset definition, and "
            "they sum to v(N)=24). Shapley phi_i = sum_{T ni i} d(T)/|T| gives (1+3+4, 2+3+4, "
            "3+0+4) = (8,9,7), matching the average over all 3!=6 permutations; Banzhaf "
            "sum d(T)/2^(|T|-1) gives (1+3+3, 2+3+3, 3+0+3) = (7,8,6), which differs from "
            "Shapley because |T|=3 exists. Interactions sum d(T)/(|T|-1) over T containing the "
            "pair: ab = 6/1+12/2 = 12, ac = bc = 12/2 = 6. All dividends are nonnegative, so "
            "every marginal is nonnegative and the table is monotone."
        ),
    },
    "features.thin_plate_spline": {
        "arguments": {
            "points": [[1.0, 2.0], [3.0, 2.0], [1.0, 4.0], [3.0, 4.0]],
            "values": [3.0, 7.0, 1.0, 5.0],
            "queries": [[5.0, 6.0]],
            "length_scale": 2.0,
            "smoothing": 0.0,
        },
        "expect": {
            "center_count": 4,
            "coordinate_origin": [1.0, 2.0],
            "length_scale": 2.0,
            "smoothing": 0.0,
            "radial_coefficients": [
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
            ],
            "affine_coefficients_constant_x_y": [
                {"value": 3.0, "underflow": False},
                {"value": 4.0, "underflow": False},
                {"value": -2.0, "underflow": False},
            ],
            "fitted_values": [
                {"value": 3.0, "underflow": False},
                {"value": 7.0, "underflow": False},
                {"value": 1.0, "underflow": False},
                {"value": 5.0, "underflow": False},
            ],
            "fit_residuals": [
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
            ],
            "maximum_absolute_fit_residual": {"value": 0.0, "underflow": False},
            "maximum_absolute_returned_system_residual": {"value": 0.0, "underflow": False},
            "returned_affine_moment_residuals": [
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
                {"value": 0.0, "underflow": False},
            ],
            "maximum_absolute_coefficient_rounding": {"value": 0.0, "underflow": False},
            "queries": [
                {
                    "source_row": 0,
                    "value": {"value": 7.0, "underflow": False},
                    "coordinate_gradient": [
                        {"value": 2.0, "underflow": False},
                        {"value": -1.0, "underflow": False},
                    ],
                }
            ],
            "kernel_evaluations": 10,
            "logarithm_underflow_count": 0,
            "kernel_underflow_count": 0,
            "exact_rounded_kernel_system_solve_checked": True,
            "evaluations_use_returned_coefficients": True,
            "transcendental_kernel_exact": False,
            "conditioning_or_error_bound_verified": False,
        },
        "approx": False,
        "note": (
            "The data are exactly affine, f(x,y)=3+2x-y, so the thin-plate interpolant must have "
            "zero radial part: with origin (1,2) and scale 2 the normalized centers are the unit "
            "square and f becomes 3+4*nx-2*ny, giving affine tail (3,4,-2). w=0 satisfies the "
            "moment rows P'w=0 and P a reproduces the data exactly, and the 7x7 saddle matrix is "
            "nonsingular (ker P' is spanned by (1,-1,-1,1), on which w'Kw = 4*log 2 != 0), so "
            "exact elimination returns that unique solution and every residual is 0. numpy's "
            "float solve of the same system returns [0,0,0,0,3,4,-2] and scipy's "
            "RBFInterpolator(thin_plate_spline) reproduces the four values and gives 7.0 at the "
            "extrapolated (5,6). Gradient = (4,-2)/scale = (2,-1) = grad f. Kernels: C(4,2)=6 "
            "assembly calls plus 4 per query = 10, none underflowing."
        ),
    },
}
