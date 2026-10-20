# Ungated epoch acknowledgments (owner review)

Status: owner review. These acknowledgments are **not** reverted.
They landed while push CI gated only `github.event.before`..HEAD, so a
run that was cancelled or superseded while pending never admission-gated,
ack-gated, or chain-gated the commits it carried. ZR-9V-2 records the
last completed green evidence run as the base. Re-signing pins and
deciding whether any of these acks are self-excused stays human.

The count of 39 is the ledger growth from `84d3d31b7` through
`021b79fec` inclusive (13 + 1 + 10 + 1 + 2 + 6 + 6). The parent ledger
(`c66e8e1ff36c`) held 5 acks. This list does not include later acks.

needs-stamp: adding this file touches the `docs/*` chain. Do not stamp
from this change. Do not edit `quality/epoch_heads.json` here.

## 84d3d31b7 (13)

`84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` — fix(integrity): reseal lane receipts and repair quality epoch fork

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 1 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `57ad3394513fe58e48127141858a990580fade56ae4755ad0a3b868aa2568f2b` | `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128` |
| 2 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `c95ce361609bdb00912147e0f87548fdbd2c0221ad27b8c1a4d0ba558b9330e1` | `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af` |
| 3 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `1a003a670945452a5c86d033900bc33429f2981422c8938b1b8e2469fc764f3a` | `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7` |
| 4 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `3ce1cd7f59c301be6b4710326ab905313725317bee3669ae8e755f6d422bc12d` | `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03` |
| 5 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `e48fce1ae4cebf94e417773bdef20c9ed7b0822397505c6a51e8843a8da06c81` | `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6` |
| 6 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_05bbdbfc2bab9f19.json` | mutated | `b9b70f7186d75054e91ee734b87b8aeeba22d66af713806a8d5410a9bc603ee3` | `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241` |
| 7 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128` | `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f` |
| 8 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af` | `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f` |
| 9 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7` | `ed1736ed83c33d0c44775292c3e4979aaf11bb5bd2d18cfeb7df3baa78575a84` |
| 10 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03` | `82c474e25984e9418b6709f81dd34e55457c512c6eef7d3b3275c249e2e82b38` |
| 11 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6` | `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06` |
| 12 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_6f6d75cd3751c48d.json` | mutated | `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241` | `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453` |
| 13 | `quality/*.json` | `corpus_epoch_cd4b7ee4ad32d328.json` | `corpus_epoch_792426f656af6eb2.json` | removed | `4a428c8f6834cbbd495e55115f50a398bfec7b3b6a1f04ba4061951288a60a7e` | `` |

Reasons:

- `capacity_eval_3622d0c059d42009.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `fleet_eval_5e907be710811a44.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `honest_verdict_b133e8b3992af893.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `monitor_run_13d20aef00774027.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `rankic_eval_8d740c0ddf6d5c54.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `vol_bench_f3bf2afc2f159ace.json`: Replay artifact path+digest refresh (faafaf040); seal updated in follow-up.
- `capacity_eval_3622d0c059d42009.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `fleet_eval_5e907be710811a44.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `honest_verdict_b133e8b3992af893.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `monitor_run_13d20aef00774027.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `rankic_eval_8d740c0ddf6d5c54.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `vol_bench_f3bf2afc2f159ace.json`: Reseal receipt_sha256 after replay binding update; head bytes match lane replay pins.
- `corpus_epoch_cd4b7ee4ad32d328.json`: Drop forked quality epoch tip (parallel stamp from faafaf040); canonical chain follows e31cd642→cf04.

## f327796b0 (1)

`f327796b039d15a8b3415ca42449dff4cdd90dc3` — chore(quality): drop tests epoch fork and refresh web fixtures

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 14 | `tests/*` | `corpus_epoch_05724f90cfec84f6.json` | `corpus_epoch_446ddadac2efb9d5.json` | removed | `cc3fccef8156e6e0369fd6a66da799e78fbadb656cb2350306d4840caab730fe` | `` |

Reasons:

- `corpus_epoch_05724f90cfec84f6.json`: Removed orphaned fork tip that branched from corpus_epoch_fd8b09ac86c69256.json alongside the canonical heads-pin chain through corpus_epoch_68f3489ea52c4bcf.json. Clears epoch_fork / epoch_multiple_heads on verify-repo and evidence-audit.

## fedb3e29f (10)

`fedb3e29f2158b75198e4cf725d95aadd73841a6` — chore(quality): ack historical src/scripts epoch removals

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 15 | `src/*` | `quant_fund/proof/promotion_receipt.py` | `corpus_epoch_52900236a145cedd.json` | removed | `ac59481b586ad02b3acacc9a0b96e9b1aab8e3505af3934d2cee90e61a287561` | `` |
| 16 | `src/*` | `quant_fund/schemas/receipt.py` | `corpus_epoch_52900236a145cedd.json` | removed | `fe5bbcaf7827bbb670a9feeb9cbeb07d5ea341b5f3390f601767fd4521804cd9` | `` |
| 17 | `scripts/*` | `scripts/README.md` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `ba0562b22f96e7606e26028fc862dfb07576355671d83a7afe9c6341c5659ccc` | `` |
| 18 | `scripts/*` | `scripts/benchmark_crypto_external.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `2b67812dd0f8a0409d76a9c0a7722c8c72c4e56e214ebb5a5cb52d93dcdc3206` | `` |
| 19 | `scripts/*` | `scripts/benchmark_garch.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `3ccc0eb5b9c8ab196b95f4dcb77e003a7e70d8272564074271c6ecfbdd446eb6` | `` |
| 20 | `scripts/*` | `scripts/benchmark_paper_loop.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `cbdfaa22ae5cc55d04fc9116400b1def62c0e304aae9a6b8476861235f28a725` | `` |
| 21 | `scripts/*` | `scripts/keep_awake_for_agents.sh` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `072909f25152892bdbbcbe5ac96bc002f4c3a92eb523cb2608b9e3f6da121256` | `` |
| 22 | `scripts/*` | `scripts/verify_binance_receipt.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `72a85d71f24a02143c42f5b16cd088fa6443b1d03a0157e4d3e2726ccafc5276` | `` |
| 23 | `scripts/*` | `scripts/verify_bugbot_findings.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `c191397754af2fa19289022a3c29dfd74be9e01b0754f0f57ae8ecfcb7b2aece` | `` |
| 24 | `scripts/*` | `scripts/verify_external_receipt.py` | `corpus_epoch_55b9336a9b3bc833.json` | removed | `951663c2ae6dc094cb5bc946bb28fade7d36d069192a3e195f38e90f751a772f` | `` |

Reasons:

- `quant_fund/proof/promotion_receipt.py`: Historical src member removal already sealed in corpus_epoch_52900236a145cedd.json; acknowledged so evidence-audit --epoch-acks can pass without rewriting the append-only chain.
- `quant_fund/schemas/receipt.py`: Historical src member removal already sealed in corpus_epoch_52900236a145cedd.json; acknowledged so evidence-audit --epoch-acks can pass without rewriting the append-only chain.
- `scripts/README.md`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/benchmark_crypto_external.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/benchmark_garch.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/benchmark_paper_loop.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/keep_awake_for_agents.sh`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/verify_binance_receipt.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/verify_bugbot_findings.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.
- `scripts/verify_external_receipt.py`: Script path removed/relocated at corpus_epoch_55b9336a9b3bc833.json; acknowledged for evidence-audit epoch-acks.

## adf8ba952 (1)

`adf8ba95234d73877a2f2a37e85cd09b9f30a13e` — chore(integrity): restamp corpus epochs after the membership fix

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 25 | `data/metadata/*` | `resolved_config.json` | `corpus_epoch_e73c101a7ab5d51b.json` | removed | `375ba1ea5473e0c275bb77f995fdb22a64290de93a918e1313c7250e372f9ab2` | `` |

Reasons:

- `resolved_config.json`: Gitignored derived config (data/metadata/**). It was stamped from a working tree that still had the file, so a clean CI checkout failed head_member_missing_live. The bytes were never committed. corpus_epoch_e73c101a7ab5d51b drops it; member_digests now skips gitignored untracked files.

## 1457ee3a5 (2)

`1457ee3a5c1564d624eddc827739cb33ab121beb` — chore(integrity): collapse the forked docs epoch chain

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 26 | `docs/*` | `corpus_epoch_bf34205674030d20.json` | `corpus_epoch_56ab5b59290569ce.json` | removed | `e297fe6338ab7451ae64ff0d42cd616b9dfd1752a7595ddd5629f725942c4bdf` | `` |
| 27 | `docs/*` | `corpus_epoch_6b3d85be9a477735.json` | `corpus_epoch_56ab5b59290569ce.json` | removed | `322f79a5d6cfdb0947d0c210784bd38934477a01e4b80a142cd725ec03ec4aad` | `` |

Reasons:

- `corpus_epoch_bf34205674030d20.json`: Parallel docs stamps forked at corpus_epoch_5ec65ddd85b8b6f9.json. The pinned line matches the live tree. This side-line receipt is removed so the chain has one head. The bytes remain in git history.
- `corpus_epoch_6b3d85be9a477735.json`: Continuation of the unpinned docs fork. Removed with that side line so check_epoch_chain sees one head. The bytes remain in git history.

## 2b72e0501 (6)

`2b72e0501897571117175c41916052b8aa078b95` — fix(integrity): bind seal-prefixed lane receipts and yml epoch extension

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 28 | `receipts/*.json` | `capacity_eval_3622d0c059d42009.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f` | `` |
| 29 | `receipts/*.json` | `fleet_eval_5e907be710811a44.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f` | `` |
| 30 | `receipts/*.json` | `honest_verdict_b133e8b3992af893.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `ed1736ed83c33d0c44775292c3e4979aaf11bb5bd2d18cfeb7df3baa78575a84` | `` |
| 31 | `receipts/*.json` | `monitor_run_13d20aef00774027.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `82c474e25984e9418b6709f81dd34e55457c512c6eef7d3b3275c249e2e82b38` | `` |
| 32 | `receipts/*.json` | `rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06` | `` |
| 33 | `receipts/*.json` | `vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_1d1a6cfdadf37c0c.json` | removed | `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453` | `` |

Reasons:

- `capacity_eval_3622d0c059d42009.json`: Seal-prefix rename to capacity_eval_5cf0042e91c7953c.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.
- `fleet_eval_5e907be710811a44.json`: Seal-prefix rename to fleet_eval_ab613c423b36627d.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.
- `honest_verdict_b133e8b3992af893.json`: Seal-prefix rename to honest_verdict_9d631fa6aace437a.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.
- `monitor_run_13d20aef00774027.json`: Seal-prefix rename to monitor_run_e69a3aed9568e30d.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.
- `rankic_eval_8d740c0ddf6d5c54.json`: Seal-prefix rename to rankic_eval_addd924f107eedd8.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.
- `vol_bench_f3bf2afc2f159ace.json`: Seal-prefix rename to vol_bench_dc16af46c3f2975b.json at corpus_epoch_1d1a6cfdadf37c0c.json. 84d3d31b7 refreshed receipt_sha256 in place after artifact-binding drift, so the filename no longer matched the seal. Bytes are unchanged by the rename; the new name is the seal prefix.

## 021b79fec (6)

`021b79fec5624774a5debc833c35fc59da18d5e0` — chore(integrity): export seal-prefixed fixtures and re-pin jewels

| # | corpus | member | epoch | transition | from_sha256 | to_sha256 |
|---|---|---|---|---|---|---|
| 34 | `web/*` | `public/fixtures/receipts/capacity_eval_3622d0c059d42009.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f` | `` |
| 35 | `web/*` | `public/fixtures/receipts/fleet_eval_5e907be710811a44.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f` | `` |
| 36 | `web/*` | `public/fixtures/receipts/honest_verdict_b133e8b3992af893.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `ed1736ed83c33d0c44775292c3e4979aaf11bb5bd2d18cfeb7df3baa78575a84` | `` |
| 37 | `web/*` | `public/fixtures/receipts/monitor_run_13d20aef00774027.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `82c474e25984e9418b6709f81dd34e55457c512c6eef7d3b3275c249e2e82b38` | `` |
| 38 | `web/*` | `public/fixtures/receipts/rankic_eval_8d740c0ddf6d5c54.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06` | `` |
| 39 | `web/*` | `public/fixtures/receipts/vol_bench_f3bf2afc2f159ace.json` | `corpus_epoch_12bce57f7c47989b.json` | removed | `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453` | `` |

Reasons:

- `public/fixtures/receipts/capacity_eval_3622d0c059d42009.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.
- `public/fixtures/receipts/fleet_eval_5e907be710811a44.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.
- `public/fixtures/receipts/honest_verdict_b133e8b3992af893.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.
- `public/fixtures/receipts/monitor_run_13d20aef00774027.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.
- `public/fixtures/receipts/rankic_eval_8d740c0ddf6d5c54.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.
- `public/fixtures/receipts/vol_bench_f3bf2afc2f159ace.json`: Fixture copy of a lane receipt renamed onto its seal prefix. Removed at corpus_epoch_12bce57f7c47989b.json; the new fixture name is the seal-prefixed receipt.

## Later ack, not part of the 39

`761ec7803e57bff282e216e4c87d612610292ddf` added one further tests removal after this series.
It is listed so the owner does not think the ledger stopped at 44.
It is not reverted either.

- corpus `tests/*` member `formal/_test_sm_probe.py` epoch `corpus_epoch_3f781cccc233472c.json`
  from `a6d832b8bc571ba9b621df31e843491d5ae62de847f46274727021c2bf5a9385` transition removed.
  Named by the tests head corpus_epoch_8285395572fc3f8f.json stamped in cf6b3613b, and absent from every git commit (git log --all -- tests/formal/_test_sm_probe.py is empty). The restamp corpus_epoch_3f781cccc233472c.json drops it so the head matches the committed tree. The bytes were never committed; this acknowledgment records that removal and does not restore them.

