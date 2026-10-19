# D1-08 stamped receipts that would fail admission honesty

Owner decides register / re-admit / grandfather. This file does not grandfather.

**Measured at:** `b5f1f6d11` (receipts head unchanged).

**Command:** load `quality/epoch_heads.json` → receipts head `corpus_epoch_1d1a6cfdadf37c0c.json`; take stamped members; drop `corpus_epoch_*` names and paths in `quality/legacy_quarantine.json`; run `honesty_stamp_errors` (`quant_fund.research.legacy_unsealed`).

Work-order figure was 176 of 346 with 6 registered. Live measurement:

- stamped members: 403
- epoch receipts (`corpus_epoch_*`): 48
- claim members: 355
- claim members also in quarantine registry: 17
- **stamped non-legacy claim receipts: 338**
- **would fail admission honesty: 170**
- would reject (`live_pnl_claim: true` / `research_only: false`): 0
- non-legacy that already carry the stamps: 168
- `admission_honesty_gap` registry entries: 1 (`p0_evidence_inventory_v1.json` — not in the 338, already quarantined)
- committed `receipt_admission.v1` receipts: 1 (`receipt_admission_529da640b520a49d.json`)

None of the 170 honesty-gapped stamped claim receipts are in the quarantine registry. Registering them as `admission_honesty_gap` is an owner decision (and `quarantine-gate` only allows new entries for unstamped base bytes).

## Gap combinations

- 143: `['live_pnl_claim_missing']`
- 10: `['research_only_missing']`
- 6: `['research_only_missing', 'data_label_missing']`
- 5: `['research_only_missing', 'live_pnl_claim_missing']`
- 4: `['data_label_missing']`
- 2: `['research_only_missing', 'live_pnl_claim_missing', 'data_label_missing']`

Machine-readable copy: [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.json`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.json).

## Receipts

| receipt | schema | kind | gaps | sha256 |
|---------|--------|------|------|--------|
| `abc_calibrate_amzn.json` | `abc_calibrate.v1` | `abc_calibrate` | `live_pnl_claim_missing` | `31aedff635330e6727b42bfa6b0c406c24aa305825c41cb09ae6aac33dca65f1` |
| `aftermath_flow_amzn.json` | `aftermath_flow.v1` | `aftermath_flow_bench` | `live_pnl_claim_missing` | `229bfe14cfd239f30786ef89c8aa0f2b2775800e00cd85a6a9d3239265604da8` |
| `anchor_scan_amzn.json` | `anchor_scan.v1` | `anchor_scan_bench` | `live_pnl_claim_missing` | `65c11ec3e7f771bbedf4cde31b63bbe9576e97a9de55f7ab030d3c8c34c28c8a` |
| `api_fuzz.json` | `api_fuzz.v1` | `api_fuzz` | `live_pnl_claim_missing` | `c403f16743569dd1a6b7aa0711a58a42ca8bdd26abc5b0645bf5ab19952e792e` |
| `band_occupancy.json` | `band_occupancy.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `98635a72af8ee0b3f395cefb81bc3be5018d39b7f9cdd8a359007c50780e70da` |
| `band_shape.json` | `band_shape.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `1e0d626f6a4b69a5761dc498a5b1bcd7d446b7a0d6c5439bf6c3beb048eafcbc` |
| `cancel_cluster_amzn.json` | `cancel_cluster.v1` | `cancel_cluster` | `live_pnl_claim_missing` | `d6554e87ef5ca20ee1df623ac0b6f08fba4b91f2c00bce68bfa949767691abf1` |
| `cancel_gradient_amzn.json` | `cancel_gradient.v1` | `cancel_gradient` | `live_pnl_claim_missing` | `b140be66a1d4467c73f104a992ab9f692ed83b959a7918461b3951fd091bff04` |
| `cancel_gradient_bench.json` | `cancel_gradient_bench.v1` | `cancel_gradient_bench` | `live_pnl_claim_missing` | `d01943ba6fa197eb7b8f259b796acc28a6cde10a2bee9995dd47a98b0535a91e` |
| `cancel_lead_amzn.json` | `cancel_lead.v1` | `cancel_lead` | `live_pnl_claim_missing` | `8151e7dc1cbbedb112d8fdc3cd3e0f52aca58ef4d23ad698d2229a02060a705e` |
| `capacity_eval_5cf0042e91c7953c.json` | `capacity_overlay.v1` | `capacity_overlay_eval` | `research_only_missing` | `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f` |
| `capacity_eval_cd0854242ed8a9ec.json` | `capacity_overlay.v1` | `capacity_overlay_eval` | `research_only_missing` | `e0a6c8250891cd5065edfc3bb385e6fdd640e5ca376cac496420815ab16881ed` |
| `churn_reseed.json` | `churn_reseed.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `3a444c77bfef84e0e070d4611368569639ee6a551ed30cbdc6a107202ccbff7d` |
| `churn_stability.json` | `churn_stability.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `7ab02a6f41f6c626d79bca729a02300d4c79dae243758a460b57a636604fdcc4` |
| `closure_fit_amzn.json` | `closure_fit.v1` | `closure_fit_bench` | `live_pnl_claim_missing` | `ff6d30e8251032109ace27be9dd1e488206be1eeb81d5fa1f9c5d8f56eb29294` |
| `closure_stack_amzn.json` | `closure_stack.v1` | `closure_stack_bench` | `live_pnl_claim_missing` | `994f8f2cf89faaace83e6e3f7441d3361e111c0d2b9e05cf123c2c602c2389cd` |
| `coherence_2dd641ab766a536a.json` | `receipt.v2` | `coherence_eval` | `research_only_missing` | `dcaf457766306077f90249a7154302297b30fc63417e2fc9a46517c71e4265fd` |
| `concordance_df424fa2f6b1c4e9.json` | `receipt.v2` | `selection_concordance` | `research_only_missing,live_pnl_claim_missing,data_label_missing` | `d35b4d196b598847e2bcb69e7b3d5536bbab0e654f94ec9a32f17554d0a7e81c` |
| `continuation_attr_amzn.json` | `continuation_attr.v1` | `bench` | `live_pnl_claim_missing` | `e2340d411b4af81d839dc85ad7326835fe7a8c97c834432c861664c773f0cbbc` |
| `coverage_cs_real_drill.json` | `coverage_cs.v1` | `coverage_cs` | `research_only_missing,live_pnl_claim_missing` | `9dfc169e6c3cb41c803c92114a4b33004d407295fc69c7a7de3dab3141a624b7` |
| `coverage_real_drill.json` | `coverage_audit.v1` | `coverage_audit` | `research_only_missing,live_pnl_claim_missing` | `2521152de17c0974039cb4f4867cda6f2ad1299e3f00135fc0764e32fd17759a` |
| `crown_behind.json` | `crown_behind.v1` | `sim_bench` | `live_pnl_claim_missing` | `1ae02ca68d023ddd18c2b0023b240d1cf729b0a2b04597954548402afbe769c6` |
| `crown_density_amzn.json` | `crown_density.v1` | `sim_bench` | `live_pnl_claim_missing` | `4945e95537a122d7d13c12d84fe59e518e3051eb8ddd9ca8db66f3218a0c6268` |
| `crown_join.json` | `crown_join.v1` | `sim_bench` | `live_pnl_claim_missing` | `df42ab245b4f82d535fb52d4a4068414b4b11537e152ec63b0e3ab6823e40af9` |
| `crown_size.json` | `crown_size.v1` | `sim_bench` | `live_pnl_claim_missing` | `751a7c4351d1639d547330d7b8f8251fb6341f500f099304a13945810a560fe0` |
| `cxl_shield_amzn.json` | `cxl_shield.v1` | `cxl_shield_bench` | `live_pnl_claim_missing` | `16cac426d34c9af016963a251e9fef5422babd2ced8be9795a7413e9c6b84e40` |
| `decay_watch_synth.json` | `decay_watch.v1` | `decay_watch` | `live_pnl_claim_missing` | `8f37ccea904fe09557819f36a60d5dcb2d99467db21301111b5c7d9836ccf6d4` |
| `deep_book_bench.json` | `deep_book_bench.v1` | `deep_book_bench` | `live_pnl_claim_missing` | `9410e500ad2602dc085c5146640d7bf7219a9626c8bccd32ea47a71829f03392` |
| `deep_microprice_amzn.json` | `deep_microprice.v1` | `deep_microprice` | `live_pnl_claim_missing` | `a9fddf27e4c26bc144ced56ca35856411f8a6220d91100872170b60411738acd` |
| `depth_consumption_amzn.json` | `depth_consumption.v1` | `depth_consumption` | `live_pnl_claim_missing` | `1bd84264cd6b886a28c16ff5246ec2c4467f81d0a656589c534dad34d8754846` |
| `depth_tilt_amzn.json` | `depth_tilt.v1` | `depth_tilt_bench` | `live_pnl_claim_missing` | `a42581086e0846cbc71a272b6cb39e018031176fd7414ebc57dc95ba7415d2b2` |
| `empirical_flow_bench.json` | `empirical_flow.v1` | `empirical_flow_bench` | `live_pnl_claim_missing` | `e12abc02ca462ff660e4aaa39fe5182508ab5896c1bdab180c5bb6caaf4c9c99` |
| `event_burst_amzn.json` | `event_burst.v1` | `event_burst` | `live_pnl_claim_missing` | `4c8872dc6f781544ef15cbace7537df2876d0320410570283e558306a9c5b019` |
| `event_granger_amzn.json` | `event_granger.v1` | `event_granger` | `live_pnl_claim_missing` | `628431d7b7099ac23e9e0c40a0b99de811952a701333949202e2ea0696256362` |
| `event_matrix_amzn.json` | `event_matrix.v1` | `event_matrix` | `live_pnl_claim_missing` | `e1a8e84ffc8b0ac08a7e8a821bc0650cb56bd10c924318e8c0be4a74587710d1` |
| `evidence_audit_3464d8f8197bf737.json` | `receipt.v2` | `evidence_audit` | `research_only_missing,data_label_missing` | `77c38c8edc534276ad901f4c64c5024f8e79c99c5ece43b001b003f362584a5d` |
| `evidence_audit_5c9f07c823727701.json` | `receipt.v2` | `evidence_audit` | `research_only_missing,data_label_missing` | `6124e972cc182ef134a5c411f2794c4f2e3fcea205f6a0bfc0a05f4ec912cd23` |
| `evidence_audit_d449e1ca0cc119a6.json` | `receipt.v2` | `evidence_audit` | `research_only_missing,data_label_missing` | `deb75d4ab1744617a0032397524891613c50282581f913d344e4acda9c6cd9d1` |
| `evidence_audit_f45ae8d6bc7cd677.json` | `receipt.v2` | `evidence_audit` | `research_only_missing,data_label_missing` | `04128fa07fd2ca3650981d0cad06d07f921a525b301945b2c539ddd8c7447989` |
| `exec_cost_real_amzn.json` | `exec_cost_real.v1` | `exec_cost_real` | `live_pnl_claim_missing` | `297d59df884a80c0503cad9c511579193cf77504fd5313492b732b4297993788` |
| `exec_cost_split.json` | `exec_cost_split.v1` | `exec_cost_split` | `live_pnl_claim_missing` | `53ef2528042134aeb7bd8c20d6c94f7e4ab3391fafd9851531f84df3fe3ae9f5` |
| `fast_replay_p42_conformance_20260928.json` | `None` | `None` | `data_label_missing` | `e8fd3b17e82fc21f39ecd184025c69b336f7be0ea18005d132a9c234e85acec6` |
| `fifo_priority_amzn.json` | `1` | `fifo_priority.v1` | `live_pnl_claim_missing` | `f180b99da62d936a31bf9df79f637141a9b7bd4c4679fc3782739455dafd7e30` |
| `flee_wide.json` | `flee_wide.v1` | `sim_bench` | `live_pnl_claim_missing` | `186fec1776108a9051fbe02f47f411279a710999045cc97028fcf5a9f9482e77` |
| `fleet_eval_5ddf15b0dc7d3ca1.json` | `fleet_eval.v1` | `distribution_fleet_eval` | `research_only_missing` | `67713fe534eb3c4a508e4c0211836026af4f9efe8c27cd2cc7d69118a9cd1b98` |
| `fleet_eval_ab613c423b36627d.json` | `fleet_eval.v1` | `distribution_fleet_eval` | `research_only_missing` | `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f` |
| `floor_compose.json` | `floor_compose.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `53d4493e5009a0da44acdab958fb3bf8df5dcc01fd5bc76d5d24fe04d07692f1` |
| `floor_pins.json` | `floor_pins.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `a38bb073ef069de28dff2ada970c05f2ac57a19433df20a19dfab727e9a0a5a0` |
| `floor_rate.json` | `floor_rate.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `632d421666dcd4ea2e7d4acd798ef82cab735054aeaa7d719eaacda91fea35f3` |
| `floor_reseed.json` | `floor_reseed.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `2c02b255e4506593755d30d448e15bdc8f242e81ecbf380056b3a8c32698060a` |
| `floor_stability.json` | `floor_stability.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `fbd9860771b933e799ee938c90ba3ea43620ead6dd99f0f44231c7c5d9916ce9` |
| `flow_couple.json` | `flow_couple.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `5dc9e98c044ada0edef8fc7f0be4ec7efabdc5d52781d9cacbd327549fb9f992` |
| `full_impact.json` | `full_impact.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `29a1b45e52fca254985f19c30d096eb6b2de14a406494282206e1092b3ea905e` |
| `full_stack.json` | `full_stack.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `2048f94b0d052bf5ebfa38c6786d60b900b37468a814ec46e9b25262d6c9d1fc` |
| `gap_close.json` | `gap_close.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `c405793974f1ca9c88feb19c56e341185c68d87809dd755eee6cc604c65a85e7` |
| `glft_bench_synth.json` | `glft_bench.v1` | `glft_bench` | `live_pnl_claim_missing` | `4efd3a9a814d245873371a8f09c2d2502fca71393dbf738496e59cb4dc369467` |
| `grammar_map.json` | `grammar_map.v1` | `wave25_map` | `live_pnl_claim_missing` | `f8ee1c31439b03bbe14989ee0cebb7b0917f5186326cc91c2d81d7705686198a` |
| `hawkes_clock_bench.json` | `hawkes_clock.v1` | `hawkes_clock_bench` | `live_pnl_claim_missing` | `816f8bd60cab106505a1adb153803ee48579ce6615a85ccd3e5d27785f2f3b77` |
| `hawkes_real_amzn.json` | `hawkes_real.v1` | `hawkes_real` | `live_pnl_claim_missing` | `9d398acb03bfc6bfd9f965d5d6115b5ca4d64244d312e90993ea11e7bd8a80ea` |
| `hidden_depth_amzn.json` | `hidden_depth.v1` | `hidden_depth` | `live_pnl_claim_missing` | `24b6c5ca511ec6211f79115fdac3eac85f52d393738a5a88adebad8c46575ea2` |
| `hidden_depth_bench.json` | `hidden_depth_bench.v1` | `hidden_depth_bench` | `live_pnl_claim_missing` | `4e18ea2564f675b351bfdd93ce9c1541af0638ef07e43728cfb48a83b3912330` |
| `hit_flee_amzn.json` | `hit_flee.v1` | `sim_bench` | `live_pnl_claim_missing` | `72765463da5af6990d81e3026dd7d23ecf83265407fb0b0b43d14a5d2690a6d0` |
| `hit_starve_amzn.json` | `hit_starve.v1` | `hit_starve_bench` | `live_pnl_claim_missing` | `3d98298e540b6ab7fa14c9b8452feb06c7b5c28a2b09a55ea8a9c4b6710acb39` |
| `hmm_verify.json` | `hmm_verify.v1` | `hmm_verify` | `live_pnl_claim_missing` | `269ee7b7e767d0388ab285e3257a5de62f4a631ada045aa83577b2b7786a51f2` |
| `ice_budget.json` | `ice_budget.v1` | `sim_bench` | `live_pnl_claim_missing` | `b0be60f820caea3783400f5f7eaff399c68d243ee11508a95212d64907789642` |
| `ice_crown.json` | `ice_crown.v1` | `sim_bench` | `live_pnl_claim_missing` | `a57b172d4c2d0f528461fd35dbcaafefc1af9b3b08c9cb2cd690d7e76053665c` |
| `iceberg_bench.json` | `iceberg.v1` | `iceberg_bench` | `live_pnl_claim_missing` | `da07cac898aadafc9bb434319334d0fc72d504fb5fe4c9ff5d0031b09499e7cc` |
| `iid_floor.json` | `iid_floor.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `cd2d6153b177482f1df86f78f6c795d03e93a17046a897c58874dc14fcde2753` |
| `imbalance_predict_amzn.json` | `imbalance_predict.v1` | `imbalance_predict` | `live_pnl_claim_missing` | `d467e13e158dddf59e0ec2dd6b0941aeae37237f491757705418f7420c789655` |
| `impact_instant_amzn.json` | `impact_instant.v1` | `impact_instant` | `live_pnl_claim_missing` | `e6d80527e2bb993d7450087db961d6229561fe1a87c0e8b68d3920ca50315c4b` |
| `impact_persist_amzn.json` | `impact_persist.v1` | `impact_persist_bench` | `live_pnl_claim_missing` | `58bcb731c697074949a3705ee2270181d9f2df5e5d2c8f08fbf04b020f3200dc` |
| `improve_flow_bench.json` | `improve_flow.v1` | `improve_flow` | `live_pnl_claim_missing` | `ec0836e7b4d24834d0b68a43c5676ea213049fce8200966dc15d98d30a0a4488` |
| `initiative_fade_bench.json` | `initiative_fade.v1` | `initiative_fade` | `live_pnl_claim_missing` | `f9a3f118c2fd37a308bd691633c7e7a9668e770d959535ca16bc2c583e4db867` |
| `instant_decomp_amzn.json` | `instant_decomp.v1` | `instant_decomp_bench` | `live_pnl_claim_missing` | `e9fcb9c72331341fe60237cb457fa8456dcbdb340e87060fbb9318a7265b4df0` |
| `intraday_exec_amzn.json` | `intraday_exec.v1` | `intraday_exec` | `live_pnl_claim_missing` | `04803c1f783145e75ccf403990146768490e123bc82c26dbdcef3901eb59fa0a` |
| `intraday_shape_amzn.json` | `intraday_shape.v1` | `intraday_shape` | `live_pnl_claim_missing` | `7e831325e8e77d61360d07100a35234c210f79b99825d6d4040d5d903470cd1d` |
| `joint_fit_amzn.json` | `joint_fit.v1` | `joint_fit_bench` | `live_pnl_claim_missing` | `1041b3196b85319688e89a359401fadef29a4ff14a7371e389668b0c2d6175bf` |
| `joint_stability.json` | `joint_stability.v1` | `sim_bench` | `live_pnl_claim_missing` | `cd43e7ec5f39ddbb3ff5c6ea9acce41191bcd250cff7e986c20637c8719c9e84` |
| `joint_tune.json` | `joint_tune.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `27aea6f04b845ae84445bcd97dbf31c6eb68cb85053af2d568fafe51027f38f6` |
| `lane_power_drill.json` | `lane_power.v1` | `lane_power` | `research_only_missing,live_pnl_claim_missing,data_label_missing` | `f0d9effae801a0ab06dfad9cbd222c55a45c71128f01b27b060c20e5624e273e` |
| `lattice_drill_vol_bench_a.json` | `vol_bench.v1` | `vol_bench` | `research_only_missing` | `33f9defac50df4f692db60eacc585c3a93eb06b7039e3be866984cee02c28481` |
| `lattice_drill_vol_bench_b.json` | `vol_bench.v1` | `vol_bench` | `research_only_missing` | `49651398e997ff6685e1ef31a1686e3799cb20c12595a8234b47c1a187990988` |
| `level_gap_amzn.json` | `level_gap.v1` | `level_gap_bench` | `live_pnl_claim_missing` | `3c1b21418c1b617164a629f45f039f341ce9bab5321c81d77831a339222cde9d` |
| `lo_response_amzn.json` | `lo_response.v1` | `lo_response_bench` | `live_pnl_claim_missing` | `df91141443b114f39d4dba23cb499de6c4c8961eb6d1d6b1b60ca6d751dd66c2` |
| `lob_exec_amzn.json` | `lob_exec.v1` | `lob_exec` | `live_pnl_claim_missing` | `e334cb33cf73a06b99bbfc834825e409e849ec860b8533a093dfefbdac2e0932` |
| `lob_invariants_amzn.json` | `lob_invariants.v1` | `lob_invariants` | `live_pnl_claim_missing` | `8a890214de32880ca3043d8ebb863992b938708828aeb3639fa79fd90e7948e0` |
| `lob_resilience_amzn.json` | `lob_resilience.v1` | `lob_resilience` | `live_pnl_claim_missing` | `b8728dbb705d8ae62e1339a4d7482070f7f827209c20c342b31ce6a9980a585d` |
| `lobster_replay_amzn_2012-06-21.json` | `lobster_replay.v1` | `lobster_replay` | `live_pnl_claim_missing` | `44f53ca3f6e42bad9617d284fc37542ca64d3b4373642fe54deae83302a9dd10` |
| `loss_cs_real_drill_conf_t_vs_empirical.json` | `loss_cs.v1` | `loss_cs.v1` | `research_only_missing,live_pnl_claim_missing` | `1859b044469b2c4ca7f0db5ed195aeb56868ef9aea0f46b5f763cf8301591763` |
| `loss_cs_real_drill_gaussian_vs_conf_t.json` | `loss_cs.v1` | `loss_cs.v1` | `research_only_missing,live_pnl_claim_missing` | `60e68e2af9cd2544458a8ac821e1468b47b322798cb498a3e116d3f0ce0964de` |
| `maker_age_bench.json` | `maker_age.v1` | `maker_age` | `live_pnl_claim_missing` | `743ba4eb58ce5575b1334ea8a1731ae39ebeab7789ea43733256e1458c8f5c49` |
| `map_parity.json` | `map_parity.v1` | `map_parity` | `live_pnl_claim_missing` | `4dff84d32892f41c5d10b30c4c55ab5ba9a072affe9f67d2c3e4c50cc39330f6` |
| `marketable_limit_amzn.json` | `marketable_limit.v1` | `marketable_limit` | `live_pnl_claim_missing` | `0e231a062d7253b3cdce1acd4a6ca72c75890e7b19fbbc47c76cc6815448c932` |
| `markout_amzn.json` | `markout.v1` | `markout` | `live_pnl_claim_missing` | `70730f6f3bd4e398497f1e302f3b375c739a548b856b68f2282f4fa18b3de864` |
| `meta_model_synth.json` | `meta_model.v1` | `meta_model` | `live_pnl_claim_missing` | `351b5285fcd9f08a8dcfa478db62ed1dfd407c9bda7065efdf2cc2fcf4e3b0e2` |
| `metaorder_detect_amzn.json` | `metaorder_detect.v1` | `metaorder_detect` | `live_pnl_claim_missing` | `c2bfd7554d2aba52493daa287850c08aced9893f7beb8f0aa976e9b0c791ec8d` |
| `mid_jump_amzn.json` | `mid_jump.v1` | `mid_jump` | `live_pnl_claim_missing` | `b4afddf6927e2254d1e0362517bdd14ec3dd07d1dcebcfbb2da40384a1585d17` |
| `mortal_repost.json` | `mortal_repost.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `c351dc60be497d864fc2cea3a06acfe8fcf3c9ef019836f3af0a0a6d58cbffd7` |
| `multih_fleet_eval_5db1cab214e291d7.json` | `receipt.v2` | `multih_fleet_eval` | `research_only_missing,data_label_missing` | `07cd44dd6a795bc0b93e21824b5a963f9c89902f405f91ce6c991e38cbdd4516` |
| `native_conformance.json` | `native_conformance.v1` | `native_conformance` | `live_pnl_claim_missing` | `6754767a42cdff244d3ff4d59b2366ff8655990087cf42957147e0890fbdc5ad` |
| `nautilus_conformance_7bf19a08c147547b.json` | `receipt.v2` | `nautilus_conformance` | `research_only_missing,data_label_missing` | `3a10fca306535858496b88890cab846c2bcffa9e7408980315162a6a5451f960` |
| `order_lifetime_amzn.json` | `order_lifetime.v1` | `order_lifetime` | `live_pnl_claim_missing` | `57d744ba5f963aed36a2fef198015643bf4be0855f610ab32dcc21bc83941e01` |
| `order_revision_amzn.json` | `order_revision.v1` | `order_revision` | `live_pnl_claim_missing` | `dbb7239f51f26e176f62493fea10a2e921b1a97945eb251eae5e3432badab3ce` |
| `pin_stability.json` | `pin_stability.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `cb002632c4d286eb8d3a1c9597f86b16d05bca58d12e3b06f472b25627bd61f1` |
| `place_law_amzn.json` | `place_law.v1` | `place_law_bench` | `live_pnl_claim_missing` | `a85ba039fc645e0455f7fa969be74bd40838b5c06b8893737d13352b68aef901` |
| `place_mix_amzn.json` | `place_mix.v1` | `place_mix_bench` | `live_pnl_claim_missing` | `b496d4d2cdab3e254968926073316a3e74167d24734a226aa22146565996a994` |
| `post_trade_drift_amzn.json` | `post_trade_drift.v1` | `post_trade_drift` | `live_pnl_claim_missing` | `83ad54e2add374fc7fa0333072c4ef745f6a1ed4112f36d14cefb9b23262192d` |
| `price_improvement_amzn.json` | `price_improvement.v1` | `price_improvement` | `live_pnl_claim_missing` | `870ef7c83529dc34f22ea1a7555abbbbed92f84d6d498096ae6159a4b4ef93f1` |
| `promotion_gate_conf_t.json` | `promotion_gate.v1` | `promotion_gate` | `live_pnl_claim_missing` | `a353477ea44ae58d507ff43f5be735937b0e0f3b2f84ef10d2d7f29e846f3506` |
| `propagator_real_amzn.json` | `propagator_real.v1` | `propagator_real` | `live_pnl_claim_missing` | `4c7a02f864596c226f78948ee5d4ae227a2323a5aee8e1efd6b9f492f3d3089d` |
| `queue_class_bench.json` | `queue_class.v1` | `queue_class` | `live_pnl_claim_missing` | `d3941977a98261a06825573ad173a9b5d21e1515ea7a3dd48430e61779dc341d` |
| `queue_fate_amzn.json` | `queue_fate.v1` | `queue_fate` | `live_pnl_claim_missing` | `54ce63133d050866a22b42879797e56d076c3f9de8f646902e2090287acedf55` |
| `queue_jump_amzn.json` | `queue_jump.v1` | `queue_jump` | `live_pnl_claim_missing` | `467134b64f8e9b397acd3f221e716b735ee62a5e70e2a533875be62b5d42924a` |
| `queue_occupancy_amzn.json` | `queue_occupancy.v1` | `queue_occupancy` | `live_pnl_claim_missing` | `c2f2b5caf9a82f3066a34abab1b0529acf44948abd7e61d7b3e1a35e20efbe23` |
| `quote_floor.json` | `quote_floor.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `20fe58eafd0338916ce1f2872ae0279f079f399ab211c61122e0be68294e1e48` |
| `quote_place_amzn.json` | `quote_place.v1` | `quote_place` | `live_pnl_claim_missing` | `493540278637dde9c9500d180d3651a9a34efa279eda7278da2a6655f5e1d4df` |
| `rankic_eval_9ebdad7da83e7348.json` | `cross_sectional_rankic.v1` | `cross_sectional_rankic_eval` | `research_only_missing` | `ff139132f80998e336918077dffacfe6d7a660305d025ca635cdcbe7837251d4` |
| `rankic_eval_addd924f107eedd8.json` | `cross_sectional_rankic.v1` | `cross_sectional_rankic_eval` | `research_only_missing` | `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06` |
| `real_benchmark_us_wide_manifest.json` | `None` | `None` | `data_label_missing` | `b6e858a81744c40d82a119027400b70b46974444019913e1ea4dc15fc288fb69` |
| `real_benchmark_us_wide_test.json` | `None` | `None` | `data_label_missing` | `df39bc572dd89e636fd56346687a2de7dfed873a9e4b3c51e2d87dbcc62d955a` |
| `real_benchmark_us_wide_validation.json` | `None` | `None` | `data_label_missing` | `8c4d8dd4b5af7419dfdf11f5f15f87b437d94db7f33d1b3719842be79b22a0a3` |
| `refill_hazard_amzn.json` | `refill_hazard.v1` | `refill_hazard_bench` | `live_pnl_claim_missing` | `195c5461c91d8d5faa684d1c3b179292e1ad717fcdabe1a61f0a851dea7e016f` |
| `regime_clock_bench.json` | `regime_clock.v1` | `regime_clock_bench` | `live_pnl_claim_missing` | `55c7d545b7f2e62b50ed135b63920effa0811d198cdc2c84fe0c035157626619` |
| `release_chase_amzn.json` | `release_chase.v1` | `release_chase_bench` | `live_pnl_claim_missing` | `8305fd598f154fe8c465ad718ece1fffe297c47471096a86543b69adfd02709d` |
| `reload_gate.json` | `reload_gate.v1` | `sim_bench` | `live_pnl_claim_missing` | `98590cf3583cb8e138ee24c43324da205e4bf04df29a91467b0f69ac5fc8deda` |
| `repost_frontier.json` | `repost_frontier.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `e946c2b132bb9334ace262e6731f96c13e14ce613f68537acf7e6d207fda2dbf` |
| `repost_latency.json` | `repost_latency.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `5afe00ca1d7db9a0c2dbbbc604b5e661b139609856fa479e035e0c158c675e8a` |
| `reseed_hazard.json` | `reseed_hazard.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `0edfaeeea62f26b073232904ea7417de7373e31214cda255219d1c23c81a5054` |
| `round_lot_amzn.json` | `round_lot.v1` | `round_lot` | `live_pnl_claim_missing` | `84c474210f467bd43f8285361cb93bc840b0a6821185a64d7fad0d0036198d6c` |
| `seed_sweep_demo_amzn.json` | `1` | `seed_sweep_demo.v1` | `live_pnl_claim_missing` | `7f5be12dde7fe0a377100dd6a2623ecd9b34a9d6d95cd68954c80f5068b9539f` |
| `shield_decay_amzn.json` | `shield_decay.v1` | `bench` | `live_pnl_claim_missing` | `2e88cdc4707a8890a0dea647749561eff8fda0d0b7711ff07bd42549e15319b2` |
| `side_imbalance_amzn.json` | `side_imbalance.v1` | `side_imbalance` | `live_pnl_claim_missing` | `fdce661d9fd73f5bd47f17dfbec526e3c4079ed0ea97dd6c59f2cb2a07791a76` |
| `sign_autocorr_real_amzn.json` | `sign_autocorr_real.v1` | `sign_autocorr_real` | `live_pnl_claim_missing` | `d2b4dce996ab14b7e2e16112862b073e87294c06fe07b755942230c65ed65438` |
| `sign_predict_amzn.json` | `sign_predict.v1` | `sign_predict` | `live_pnl_claim_missing` | `fbbe98aec24936c1b3a0dff0c922a660fdb2e2419e402bf35ba30f923a7cf13f` |
| `sim_real_ledger_amzn.json` | `sim_real_ledger.v1` | `sim_real_ledger` | `live_pnl_claim_missing` | `814b4853c24bd668b2c2fdb66f6fd4eb8b9854fcc335edc0ea22dbc6d3271622` |
| `sim_sensitivity.json` | `sim_sensitivity.v1` | `sim_sensitivity` | `live_pnl_claim_missing` | `cb33e655209b155f557770674a45d14a77e523f4d964f8cea6c69644048ca140` |
| `split_flow_bench.json` | `split_flow.v1` | `split_flow_bench` | `live_pnl_claim_missing` | `37c6fe8de4fab801aa4dbb5eff23c326ecbf8d471e869d500be9451d9fb6e73f` |
| `spread_dynamics_amzn.json` | `spread_dynamics.v1` | `spread_dynamics` | `live_pnl_claim_missing` | `13bb86c5bbf5f826279c0af371790259cfdb202c8bf85c4f09155199cdd3bdce` |
| `spread_floor_bench.json` | `spread_floor.v1` | `spread_floor_bench` | `live_pnl_claim_missing` | `37861b395d98f358141ada1962c698679c80c24632baa50e7c08b32873930a6f` |
| `spread_reopen.json` | `spread_reopen.v1` | `sim_vs_real` | `live_pnl_claim_missing` | `d22bb5a384368a0d383a9cf6df4f38b40e148830e700784a96ae446354aed406` |
| `spread_response_amzn.json` | `spread_response.v1` | `spread_response` | `live_pnl_claim_missing` | `ab2a078fc13c3bebf659ea962fb21ab18e826508f0abad2fc83e1210ba3c8463` |
| `spread_response_bench.json` | `spread_response_bench.v1` | `spread_response_bench` | `live_pnl_claim_missing` | `df3253be3c3a3f43bc4e4cccbda6d72f2a627d95ba99a683117cd9d19e9d5352` |
| `stack_watch_synth.json` | `stack_watch.v1` | `stack_watch` | `live_pnl_claim_missing` | `d90dfbd42266b54ddeb8e237e12d674ecbb24359287166f074d6241ad61a7dba` |
| `stale_quote_amzn.json` | `stale_quote.v1` | `stale_quote` | `live_pnl_claim_missing` | `91bfc70fb7cd64be078a327fae0df159395301ce2d42d954157c6749628bf6ce` |
| `streak_calibrate_bench.json` | `streak_calibrate.v1` | `streak_calibrate_bench` | `live_pnl_claim_missing` | `684c6584fc9d7d18d8e6801934e63a1ced7249321e13bc8f06f7868e60a2b86f` |
| `streak_stats_amzn.json` | `streak_stats.v1` | `streak_stats` | `live_pnl_claim_missing` | `d51d2a74e86e484183fa2af0405ad38af8273f38fb3621f2bcd39e6029556129` |
| `sweep_crown.json` | `sweep_crown.v1` | `sim_bench` | `live_pnl_claim_missing` | `d42626eead90475119af14f9c8d60d0ca5b94f62ad14c24ff157b4505f264b13` |
| `sweep_width_amzn.json` | `1` | `sweep_width.v1` | `live_pnl_claim_missing` | `ed310e546e4b4f2c2544f99c24cf010197ae78d3fa753bede0af328dde59ccdd` |
| `sweep_width_bench.json` | `sweep_width_bench.v1` | `sweep_width_bench` | `live_pnl_claim_missing` | `67f57afee23c89322c27d64070da7731ba292703388bb68768afc34019f69431` |
| `tail_real_drill.json` | `tail_audit.v1` | `tail_audit` | `research_only_missing,live_pnl_claim_missing` | `9b36d560170c53b02f873de648f4c9ea83d7e30e61ab0ad598a2ea4d05b477e4` |
| `tape_digest_amzn.json` | `tape_digest.v1` | `tape_digest` | `live_pnl_claim_missing` | `4915ad1bb6cf3c0857a09020ca27f3c7e6685a9f64681c2fefd8f2e91a8b15cd` |
| `tape_surgery_amzn.json` | `tape_surgery.v1` | `tape_surgery` | `live_pnl_claim_missing` | `4696227f39dac4b343da3dc5f9a708069d44b64db94a7f5ca12e6455c54629b5` |
| `tick_rule_amzn.json` | `tick_rule.v1` | `tick_rule` | `live_pnl_claim_missing` | `d69bc89702def00c3abee7c06e071378f18f0996a186a296a68ba16d676e3840` |
| `touch_empty.json` | `touch_empty.v1` | `sim_bench` | `live_pnl_claim_missing` | `909b434110fae047594dfa08b7342959281df4102e3ee8ba4d65d9c547319c95` |
| `touch_follow_amzn.json` | `touch_follow.v1` | `sim_bench` | `live_pnl_claim_missing` | `8dc68233094f788f38f52a8e32fc5a04a25c27c35806b73d5b677f7160033bdb` |
| `trade_decomp_amzn.json` | `trade_decomp.v1` | `trade_decomp` | `live_pnl_claim_missing` | `af272bd9ef0a1845d901080caa9c3a27257a3ae341848cdffadd8f03475c0427` |
| `unhit_chase_amzn.json` | `unhit_chase.v1` | `unhit_chase_bench` | `live_pnl_claim_missing` | `a7c9de93641c46d2853515894aa636286ba5d49bc12833aa936cc4c056080081` |
| `vac_chase_amzn.json` | `vac_chase.v1` | `vac_chase_bench` | `live_pnl_claim_missing` | `371db4e5f9b635abc95f30ea5e6e0b135202c998322124897d35b26f9a130b1c` |
| `vol_bench_dc16af46c3f2975b.json` | `vol_bench.v1` | `vol_bench` | `research_only_missing` | `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453` |
| `vol_signature_amzn.json` | `vol_signature.v1` | `vol_signature` | `live_pnl_claim_missing` | `66c853a1efbe2fe0a41535ef9e191eee310e5346e7057fd157a8832b3b1e3eb8` |
| `vpin_amzn.json` | `vpin.v1` | `vpin` | `live_pnl_claim_missing` | `feed66eda5cf279af3d12224eeeb7cae7ff47a998dcc767249d185f78bcd27db` |
| `warmup_spec.json` | `warmup_spec.v1` | `warmup_spec` | `live_pnl_claim_missing` | `01b3b622ed4e6d6852fd56f2122d257e1b9f31d100ae850cbea38921a38bafd6` |
| `wave23_map_amzn.json` | `wave23_map.v1` | `wave23_map` | `live_pnl_claim_missing` | `ad949f2ebda26df715cc45e1d6efef08c10e6deeac503580bc6668b61bf10ee6` |
| `wave24_map.json` | `wave24_map.v1` | `wave24_map` | `live_pnl_claim_missing` | `5c765aa1b4d50ddd0fb2f3645de5edc6c60465ae6b6c936b7133507e3a0069aa` |
| `zone_card.json` | `zone_card.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `751e3497d28c9a1f6e50d5040ec6b80a416194b0ea1b0ab2b05462c897b219d3` |
| `zone_churn.json` | `zone_churn.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `7f4b68803588fe0ddc3ee3c45698a7a2ea367f8c2bf5cad7e542fd56e91441f5` |
| `zone_embargo.json` | `zone_embargo.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `ee556605f12d88c494b6ead3a02099f569dfef7875cc21ca7ea5657983aea180` |
| `zone_map.json` | `zone_map.v1` | `wave24_map` | `live_pnl_claim_missing` | `43753aa2492ac472d22f2e221fa155d2c1393ed13201a1d9028231e15a4d68f7` |
| `zone_stability.json` | `zone_stability.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `bd695896930494a0d8da307b9088744b98abe575a6574369d929ba7ffc8a4b2f` |
| `zone_ttl.json` | `zone_ttl.v1` | `microstructure_bench` | `live_pnl_claim_missing` | `b5987b6ac2dfd8484a15f89fbbfa7dd8d6009df8dd1d4ec1098b4efe9580d722` |

