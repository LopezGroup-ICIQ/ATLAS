# Changelog

## 0.56.0 (2026-07-26)

### Docs

- **mlip**: update EquiformerV2 docstring ([#9e72db9](https://github.com/pol-sb/ATLAS/commit/9e72db90262ed47f9899ba1480e3a01a94c58ddf))
- **core**: update all docs to reflect new model-agnostic capabilities ([#0791ce4](https://github.com/pol-sb/ATLAS/commit/0791ce4930b87ac6c275155335e6755ded1ad750))
- add autodoc rst files for new backend and benchmark modules ([#c3f9b48](https://github.com/pol-sb/ATLAS/commit/c3f9b4834debce773abd1710c9357e2e0b6096e5))
- document modular MLIP backend architecture ([#5142ba4](https://github.com/pol-sb/ATLAS/commit/5142ba4ca56543c968cf50e1d3f8036827450c80))
- **gui**: update mockup image ([#969b7ee](https://github.com/pol-sb/ATLAS/commit/969b7eeb2f211244e132fb011c3490f6e12f2b42))

### Feature

- **domain_validity**: add mixed-precision training to autoencoder ([#0d698a1](https://github.com/pol-sb/ATLAS/commit/0d698a19f2e169485872c11275cc6c9a9221cc3c))
- **al_loop**: aggregate extrapolation statistics across AL iterations ([#813c0b8](https://github.com/pol-sb/ATLAS/commit/813c0b899792b016cd1faceaf951ea36f0d674e5))
- **domain_validity**: aggregate and expose extrapolation statistics ([#784997d](https://github.com/pol-sb/ATLAS/commit/784997df594ecd453659d16790066a384b8ee3b5))
- **al_loop**: make committee eval conditional and add confidence disagreement check ([#54e8d14](https://github.com/pol-sb/ATLAS/commit/54e8d14ec5f0170543d73a18dfd32a46c3296b37))
- **mlip**: add Orb v3 confidence uncertainty estimation ([#11c3c9d](https://github.com/pol-sb/ATLAS/commit/11c3c9d379113d35b4a4fbe8f44e8fc366308f72))
- **mlip**: add MLIPConfidenceEstimator protocol ([#d75a47e](https://github.com/pol-sb/ATLAS/commit/d75a47e268b935faa352c5ba7c61f760101fc91f))
- **core**: add Hugging Face token configuration ([#9f78279](https://github.com/pol-sb/ATLAS/commit/9f78279a2938996f1cbc414563d104cf7ffb5aaa))
- **mlip**: add descriptor extraction for Orb and EquiformerV3 backends ([#5aa76d1](https://github.com/pol-sb/ATLAS/commit/5aa76d1f7ad9f7d90a7ed4d583ad20d877cb85d0))
- **mlip**: finish fairchem/equiformer backends, md_type and fairchem_task config ([#8247795](https://github.com/pol-sb/ATLAS/commit/8247795dd1d2057cf2a60efcfe2f52eafbeb80fe))
- **mlip**: add EquiformerV3 backend for inference ([#cb7600c](https://github.com/pol-sb/ATLAS/commit/cb7600c4d2f17c3d56ddf3b967fb56aa362c6cd9))
- **mlip**: add fairchem backend for UMA/eSEN pretrained potentials ([#dffa6f0](https://github.com/pol-sb/ATLAS/commit/dffa6f0e409b4a4ceab5a999a3748dd69416df9c))
- **mlip**: add Orb backend for inference-only pretrained potentials ([#08bb0b7](https://github.com/pol-sb/ATLAS/commit/08bb0b71c0de715041b75c9e2049179ed6499331))
- **mlip**: unify foundation model resolution across backends ([#fcc5605](https://github.com/pol-sb/ATLAS/commit/fcc5605b1ddde5687cd3d33f59be37bc634b6719))
- **mlip**: add MLIPPretrainedModel protocol and MACE foundation model support ([#293ad92](https://github.com/pol-sb/ATLAS/commit/293ad926b3c9c15d2fbb5af96a35ed6fa6e3081a))
- **core**: per-backend container images ([#94eb70b](https://github.com/pol-sb/ATLAS/commit/94eb70b8af6f0ee22bd1338b6cec2724addd4e8e))
- **al_loop**: add debug mock mode for instant local AL loop testing ([#0809fca](https://github.com/pol-sb/ATLAS/commit/0809fcac23b8ebd9a23d11c9d76aace204211bde))
- **al_loop**: add morphological closing as main boundary method ([#406f221](https://github.com/pol-sb/ATLAS/commit/406f221013f35635ebd83d7027e0e995a7021146))
- **core**: implement model compilation and packaging ([#58846f4](https://github.com/pol-sb/ATLAS/commit/58846f45dfa37572c7be221fe9a0484d37bdba1c))
- **al_loop**: add model compilation infrastructure for inference ([#7ae82ab](https://github.com/pol-sb/ATLAS/commit/7ae82ab56b1b2ce88fdeed9d0eb09029d60c0eee))
- **core**: extend MLIP backend protocols with training and LAMMPS support ([#086054b](https://github.com/pol-sb/ATLAS/commit/086054b986d8ea08f6ac7796619ee829eacdb478))
- **schema**: add MLIP backend configuration options ([#87a3403](https://github.com/pol-sb/ATLAS/commit/87a34039705998b6ae37fcbf533156e4b5752681))
- **core**: add Allegro MLIP backend and generic training infrastructure ([#ba9c2fa](https://github.com/pol-sb/ATLAS/commit/ba9c2fa204c4e8f8cf0b3889e9013f2bcca04e7c))
- **core**: enhance MACE backend with foundation model and LAMMPS support ([#3fbeda1](https://github.com/pol-sb/ATLAS/commit/3fbeda1ba87180a60b015b83c3d394c097c217fc))
- **core**: register MLIP backend entry points and update utility functions ([#380bf1c](https://github.com/pol-sb/ATLAS/commit/380bf1c82c2bd4bbc71a66b2d99163a6bf846ed6))
- **core**: enable modular MLIP usage ([#977d082](https://github.com/pol-sb/ATLAS/commit/977d082a7768cf2eaa00215dc44151fa072ed76f))

### Fix

- **domain_validity**: separate lr_patience from patience ([#1fb0d43](https://github.com/pol-sb/ATLAS/commit/1fb0d43d8bff8128b6902d28bc44b372896f04c9))
- **mlip**: make orb compatible with new api ([#0d191cf](https://github.com/pol-sb/ATLAS/commit/0d191cfe77e7b79d9aaf96d6674cbe15c4f5823b))
- **dft**: fix prints and enable prepend_text in dft calcs ([#5df6037](https://github.com/pol-sb/ATLAS/commit/5df6037a05113012cdb7470b3ae6fdafd145fb9d))
- **dft**: make parallel_env optional ([#607ca11](https://github.com/pol-sb/ATLAS/commit/607ca1139d5202ba19ee8650d4aac64e3230ed44))
- **al_loop**: skip LAMMPS potential creation in mock mode ([#70944f9](https://github.com/pol-sb/ATLAS/commit/70944f9bb0a9356d58ce16583b6c534621363a42))
- **mlip**: support EquiformerV2 checkpoint ([#8ff42cf](https://github.com/pol-sb/ATLAS/commit/8ff42cfe3867d8052b3d0cac2f62df8875cdecc6))
- **md**: register Orb backend and gate committee evaluation for inference-only backends ([#75fefa6](https://github.com/pol-sb/ATLAS/commit/75fefa6f2ea4c2aeac76ceabed1c399cf3b95498))
- **mlip**: guard against inference-only backends in training path ([#1e01556](https://github.com/pol-sb/ATLAS/commit/1e015560a2c8a1fd322dea6597b6e78e389659cb))
- **md**:  unify MD calculator path and wire mlip.md_backend ([#fd83a44](https://github.com/pol-sb/ATLAS/commit/fd83a444a9ea944a8b29f97a4073e77770c90ce6))
- **mlip**: nequip import path fix ([#29326e0](https://github.com/pol-sb/ATLAS/commit/29326e0d7624a63e260e5f84b706b415d0db84e3))
- **report**: src/atlas/active_learning/report_utils.py ([#df7997f](https://github.com/pol-sb/ATLAS/commit/df7997fcd6f0a1aecd98c558fe862aad6f9cd7f5))

### Misc

- **mlip**: bump python version and make mace optional install compatible with orb ([#9afe677](https://github.com/pol-sb/ATLAS/commit/9afe67760c332040240638a3395424a025226583))
- **core**: update mlip commit scope ([#ef881d1](https://github.com/pol-sb/ATLAS/commit/ef881d16888ec6de892d54fef941650a771a527c))
- **mlip**: remove legacy scripts directory ([#32a01ff](https://github.com/pol-sb/ATLAS/commit/32a01fff7b5b69866ba6f40bf7cb09dcca1ec80b))
- **mlip**: add commit scope for mlips and backends ([#3f6df09](https://github.com/pol-sb/ATLAS/commit/3f6df093df12eab7967835e6a1bf399afecb8d59))
- **core**: upgrade torch dependencies and remove duplicates ([#6d56a74](https://github.com/pol-sb/ATLAS/commit/6d56a74cac6a891e41c000217b88ad03bd25c04a))
- **core**: update uv.lock ([#092175e](https://github.com/pol-sb/ATLAS/commit/092175eda84386c8a3222cfdf180a691b2d773cf))
- **core**: update requirements.txt ([#68bc0e1](https://github.com/pol-sb/ATLAS/commit/68bc0e1bef54d0750f3934b6c5a269729b89edd6))
- **deps**: bump gitpython from 3.1.44 to 3.1.50 ([#7aaaeee](https://github.com/pol-sb/ATLAS/commit/7aaaeeef43a91cbf199d5ba4b9b2d96d40fb2455))
- **deps**: bump gitpython from 3.1.44 to 3.1.50 ([#46a3094](https://github.com/pol-sb/ATLAS/commit/46a3094e6647f713befa64f786432a4efff4c3d9))
- **deps**: bump urllib3 from 2.3.0 to 2.7.0 ([#3892087](https://github.com/pol-sb/ATLAS/commit/3892087f3a07cd0a39fd036d5de261c6e84326a7))
- **deps**: bump urllib3 from 2.3.0 to 2.7.0 ([#0375e77](https://github.com/pol-sb/ATLAS/commit/0375e775d5beddbb169fd7e2c0953759f2bdaf3e))
- **deps**: bump pymysql from 0.9.3 to 1.1.1 ([#78640b0](https://github.com/pol-sb/ATLAS/commit/78640b0dab8078ac47611c67c4932977cb86caa8))
- **deps**: bump pymysql from 0.9.3 to 1.1.1 ([#1bb5c01](https://github.com/pol-sb/ATLAS/commit/1bb5c01a9fea5c75f4e79a2e81f3cb339b85a23a))
- **core**: address dependabot vulns. ([#2d35bc9](https://github.com/pol-sb/ATLAS/commit/2d35bc9aa2537cb7cb3749fa2ed347327e98ba36))

### Refactor

- **mlip**: extract build_orb_forcefield and add shared descriptor assembly helper ([#cb1918f](https://github.com/pol-sb/ATLAS/commit/cb1918f39b92ddb33ba700236280e41d92a664bf))
- **benchmarks**: unify foundation model route via backend registry ([#2fa6f2e](https://github.com/pol-sb/ATLAS/commit/2fa6f2e45b358351fc3cd353461ffd8d721088df))
- **mlip**: simplify MACE descriptors by routing through create_mace_calculator ([#b229eff](https://github.com/pol-sb/ATLAS/commit/b229effe90b7cd89e71eb91868ee7eb7a129fc64))
- **core**: clean-up and reformat ([#ae6acd0](https://github.com/pol-sb/ATLAS/commit/ae6acd03128f063bb4212db4db6d1ea5dab57d5d))
- **core**: consolidate structure ID standardization ([#4fd41fe](https://github.com/pol-sb/ATLAS/commit/4fd41fe63fef884ef888cc581621694bd4bf75fe))
- **core**: preserve model file extensions in CalcJobs ([#fadcf4b](https://github.com/pol-sb/ATLAS/commit/fadcf4b2d2c85b4ef962fad9b7a409310ea0599c))
- **mlip**: use inference model lookup across MD, eval, and safeguard ([#78f7ee3](https://github.com/pol-sb/ATLAS/commit/78f7ee379b76b5c3e003b9f245561ac0cd151cc9))
- **core**: cache code nodes and decouple prepend_text and add model compilation step and safeguard compilation ([#f695d40](https://github.com/pol-sb/ATLAS/commit/f695d40cb8b3a7eacfa272e026cea08d850b299d))
- **gui**: update GUI and CLI for multi-backend active learning ([#82783e4](https://github.com/pol-sb/ATLAS/commit/82783e4d2cfb3d2cf95d31297deb409230596bce))
- **safeguard**: update safeguard and reports for multi-backend support ([#0987c77](https://github.com/pol-sb/ATLAS/commit/0987c77f7661a26a1d53486cd3b0c7eb601e1b3b))
- **al_loop**: adapt active learning workflow to multi-backend support ([#a6c1090](https://github.com/pol-sb/ATLAS/commit/a6c109023c7c5fb7decba5dbd5d91bdc03077d8c))
- **core**: linting ([#043bf8b](https://github.com/pol-sb/ATLAS/commit/043bf8ba71a85d74fe7aaa7dce72c6def1980cea))

### Style

- **mlip**: fix typo ([#dd13ec7](https://github.com/pol-sb/ATLAS/commit/dd13ec7bc126fc641a665013def022c2fcf0aa10))

### Test

- **core**: mirror extrapolation statistics in mock MD process ([#e7df46a](https://github.com/pol-sb/ATLAS/commit/e7df46ac81724f7daa797e164316813260d4d219))
- **mlip**: add confidence estimator conformance and live Orb v3 tests ([#a99a0e8](https://github.com/pol-sb/ATLAS/commit/a99a0e8eb69934006b0af538d9fe5b17339c8bff))
- **al_loop**: add portable mock AL-loop integration test ([#032d85a](https://github.com/pol-sb/ATLAS/commit/032d85a51f7ac8f90f4a6fcbf958a1ab8658ea32))
- **core**: add fairchem/equiformer backend tests ([#ef1dd7e](https://github.com/pol-sb/ATLAS/commit/ef1dd7e4f96e3d3361319516d08af1665b02a5f9))
- **core**: add tests for pretrained protocol, spec parsing, and trainer guards ([#3d34de4](https://github.com/pol-sb/ATLAS/commit/3d34de4e0bd691d06a3071fdf04f1e5bafb23eb5))
- add backend protocol and integration tests ([#76afefa](https://github.com/pol-sb/ATLAS/commit/76afefa71800cf2a69732271ca8b079a00581511))

## 0.55.2 (2026-06-23)

### Docs

- **gui**: update wizard instructions ([#7eff4ec](https://github.com/pol-sb/ATLAS/commit/7eff4ec1a2fe09cf694e7df7d1baa9eb9b2ad91d))
- **gui**: add gui section to README.md ([#f430503](https://github.com/pol-sb/ATLAS/commit/f430503348a7f370ecc0587c2492a83d360d9eb4))

### Fix

- **gui**: avoid ticklabels recursion ([#10b066d](https://github.com/pol-sb/ATLAS/commit/10b066de7c9ce8190fc32f3b6ad9cc2dfe724625))
- **gui**: update icons and app id ([#d9cdaa1](https://github.com/pol-sb/ATLAS/commit/d9cdaa1ff8e86edb4a1e258080b17d3fac8e4938))

### Refactor

- **gui**: update icons ([#b464c1e](https://github.com/pol-sb/ATLAS/commit/b464c1efcb993c8582ccaf49a4a1c0c95ed22b64))

## 0.55.1 (2026-06-23)

### Fix

- **gui**: numerical sorting in table view ([#8021a84](https://github.com/pol-sb/ATLAS/commit/8021a84113ca1e202e8b76fb0b1fff9ec3631bab))
- **gui**: increase tooltip size to avoid clipping ([#b16963d](https://github.com/pol-sb/ATLAS/commit/b16963dc8f0ed9ecbb927dfb031547f1899ffbfa))
- **dft**: enhance compatibility between gui and dft submission ([#bf08aa2](https://github.com/pol-sb/ATLAS/commit/bf08aa2d7103deacab76b0cdce6f4de855bc6d20))

### Misc

- **core**: update uv.lock ([#9c9bc82](https://github.com/pol-sb/ATLAS/commit/9c9bc824aa188f3b1101ce1ada46e39ec3fd0b03))
- **core**: update requirements ([#f476184](https://github.com/pol-sb/ATLAS/commit/f476184d3bd2808fd0944aca88453555aa608fd0))

## 0.55.0 (2026-06-22)

### Feature

- **gui**: add table view filters, removal and addition ([#68e57a4](https://github.com/pol-sb/ATLAS/commit/68e57a4fa98a1ccec74ca997d1d0469556e54b49))

### Fix

- **gui**: lazy load aiida codes to prevent slowdown in development machine ([#97f780b](https://github.com/pol-sb/ATLAS/commit/97f780b06add92d0a40a62464855889c837cecc0))
- **gui**: add missing message broker setup in gui setup wizard ([#47a8b01](https://github.com/pol-sb/ATLAS/commit/47a8b013e97f777e7557c107a1afdf92c2d8c371))
- **dft**: normalize structure handling to ASE Atoms and update config schemas ([#cd21e96](https://github.com/pol-sb/ATLAS/commit/cd21e96eb42ec76e3ff44522500e33e676ca319e))
- **gui**: improve gui design and theming consistency ([#8a21422](https://github.com/pol-sb/ATLAS/commit/8a214220dd89f39f4a4cb1627cf32dedc9f719e9))
- **md**: apply shapely compatibility patch ([#09c2844](https://github.com/pol-sb/ATLAS/commit/09c28448572069f2f4fee08aec2e0e7e601fa5c2))

### Refactor

- **core**: change comments and code format ([#7489bd2](https://github.com/pol-sb/ATLAS/commit/7489bd2ab1e54b8c1429941162bc71b82850bb3b))

## 0.54.3 (2026-06-20)

### Fix

- **gui**: further optimize theme switching ([#f725295](https://github.com/pol-sb/ATLAS/commit/f72529557cd42618c8140fa3ef1f237639cb9cb2))

## 0.54.2 (2026-06-20)

### Fix

- **gui**: optimize theme switching ([#76137ca](https://github.com/pol-sb/ATLAS/commit/76137ca875a07c5eaf5373c1e2f5dd2e35ec39ec))

### Misc

- **core**: update uv.lock ([#8792763](https://github.com/pol-sb/ATLAS/commit/87927634af5e868ee498188c3fda586cbb16d65d))

## 0.54.1 (2026-06-19)

### Fix

- **gui**: added workflow view, and fixes and additions to the gui ([#7a26331](https://github.com/pol-sb/ATLAS/commit/7a26331d655226cfc5f4d6f30b21199c3ea31289))
- **init_db**: prevent unphysical distances during cluster generation using spherical method ([#9e506a4](https://github.com/pol-sb/ATLAS/commit/9e506a4ec8733cf30b64aa2de56f955c2a74fcf2))
- **gui**: improve init db deletion dialog clarity ([#76d8126](https://github.com/pol-sb/ATLAS/commit/76d81268cc07d30d3c8dadb2f0861b4a2987e408))

## 0.54.0 (2026-06-18)

### Docs

- **core**: add docs for DFT benchmark ([#c2d3cbc](https://github.com/pol-sb/ATLAS/commit/c2d3cbc5eb2efbba9c0006cbcd89ff0484afa778))

### Feature

- **dft**: re-add DFT benchmark suite ([#ecf3593](https://github.com/pol-sb/ATLAS/commit/ecf35933e4b4f23b4f4d62294d9b2e0d232bbea3))

### Fix

- **gui**: improve delete dialog and toml parsing robustness ([#62d0f8e](https://github.com/pol-sb/ATLAS/commit/62d0f8e6897e903a04273e720c3ca829d55caa50))
- **init_db**: rearrange order of operations to ensure continuous structure size distribution ([#72442ee](https://github.com/pol-sb/ATLAS/commit/72442eee47a8b30e257a539eb891b4737e784190))
- **gui**: fixes to phasediagram creation and config editor ([#6737bb9](https://github.com/pol-sb/ATLAS/commit/6737bb9a706faddfdd2fa12b7c72c7567ed46498))
- **init_db**: fix phase generation in databases ([#8ec5667](https://github.com/pol-sb/ATLAS/commit/8ec5667ccdde49fa3a5579a1e1450eda2ddcdb59))
- **gui**: improve gui setup wizard ([#b216507](https://github.com/pol-sb/ATLAS/commit/b216507aa260b689da13a25f77222d0eccb8d5fd))
- **gui**: change order of DFT benchmark and labelling in side panel ([#52955c3](https://github.com/pol-sb/ATLAS/commit/52955c36413292e1c5d7255cc7d888c59e39daae))
- **gui**: gui improvements, refactors and fixes ([#1a9b247](https://github.com/pol-sb/ATLAS/commit/1a9b2472db025c7a4b032787296be503d2bbffc3))
- **core**: fix imports ([#2de7a9d](https://github.com/pol-sb/ATLAS/commit/2de7a9d168e99617bbfdc8fbf2e6b5b03b280e22))
- **al_loop**: create only a single instance of portable code ([#c31afcd](https://github.com/pol-sb/ATLAS/commit/c31afcd3d87a5c32a2c694a47465eb49f6b8545e))
- **core**: ensure compatibility of requirements ([#623c70a](https://github.com/pol-sb/ATLAS/commit/623c70aa860d75db72a90f4767c0435a52ce5a05))
- **gui**: address slowdowns on settings tab (i) ([#7640967](https://github.com/pol-sb/ATLAS/commit/7640967be4a3475bcd6bbe7d04748b8b2b79d1bb))

### Misc

- **core**: update repository-related files ([#a6cad8c](https://github.com/pol-sb/ATLAS/commit/a6cad8c1877a09e2523af7b6a4a19a2d94bf211d))

### Refactor

- **al_loop**: remove deprecated active learning workchain ([#b5f203f](https://github.com/pol-sb/ATLAS/commit/b5f203f7a607503d3e58f4a88ae3ec1d65d34a75))

## 0.53.0 (2026-06-16)

### Docs

- **core**: update readme ([#e3c125b](https://github.com/pol-sb/ATLAS/commit/e3c125b2ceb996171241d8bbb82df2ece11b9839))

### Feature

- **gui**: add improve initial setup and wizard ([#89964ae](https://github.com/pol-sb/ATLAS/commit/89964ae58c23b2f699379bdd83d30f11519c7abc))
- **gui**: add first version of gui ([#f5fde84](https://github.com/pol-sb/ATLAS/commit/f5fde8452e473a894636f389e816bf23886fa78b))
- **core**: big batch of improvements: db gen, cluster, dft, gui ([#bf063d7](https://github.com/pol-sb/ATLAS/commit/bf063d76961f97b2105e8a8d3b043d61e3f65fa1))

### Fix

- **core**: update missed references to mdb ([#3b035f2](https://github.com/pol-sb/ATLAS/commit/3b035f2c0d87210efed24852f50eff80781bef8f))

### Misc

- **core**: update .gitignore ([#8d11ef3](https://github.com/pol-sb/ATLAS/commit/8d11ef3334df38643bdbd18b2b5de8c7092452cc))
- **core**: add uv-related files ([#e536671](https://github.com/pol-sb/ATLAS/commit/e5366719d1c734a403c5e8c97cbce520297e8b09))

### Refactor

- **gui**: updated internal paths for GUI, logos and added scope in pyproject.toml ([#eb782ac](https://github.com/pol-sb/ATLAS/commit/eb782ac75270ca327883ec3040de5b100e2dfe61))

## 0.52.6 (2026-06-05)

### Docs

- **core**: update readme ([#a389742](https://github.com/pol-sb/ATLAS/commit/a389742458d400be4c6e31bf97e6e7f5abfc7d38))
- **core**: fix author links in README.md ([#9f63a5e](https://github.com/pol-sb/ATLAS/commit/9f63a5ed75f8f0453a87658e30e90730977d9451))
- **core**: update broken pyproject.toml link ([#6424d39](https://github.com/pol-sb/ATLAS/commit/6424d3972bc2c5c1bb18c216b75a5d44c0ba6890))
- **core**: update readme ([#aba07d9](https://github.com/pol-sb/ATLAS/commit/aba07d98bbc59e5a050533264a9d8aead8f363d8))
- **core**: updated broken links in README.md ([#9f42665](https://github.com/pol-sb/ATLAS/commit/9f42665a78397bb3e9e376a58211348df0f1cfbd))
- **core**: update README ([#bac6f2d](https://github.com/pol-sb/ATLAS/commit/bac6f2df0ef2eb5faa5c11ee54d73c405c67463e))

### Fix

- **core**: rename missing executables from rebrand ([#07331ee](https://github.com/pol-sb/ATLAS/commit/07331eeec340d3383e2e55fc17279254aeaff588))
- **al_loop**: In the DFT stage, update vasp-dft output information before comparing them with mace results ([#f05965f](https://github.com/pol-sb/ATLAS/commit/f05965fffe6392b19d523932a141957e5aa7d158))

## 0.52.5 (2026-06-04)

### Docs

- **core**: update main description ([#de9ebb6](https://github.com/pol-sb/ATLAS/commit/de9ebb64a484ffaee6c3b4e402ac3491ea96b122))
- **core**: update main description ([#4f06464](https://github.com/pol-sb/ATLAS/commit/4f06464d4c11c88a304da3ffc8c392ece7aa5539))

### Fix

- **core**: update all entry points from mdb- to atl- ([#04f73c4](https://github.com/pol-sb/ATLAS/commit/04f73c4d1b746473a10bacaf489a77b34c8b98bd))

## 0.52.4 (2026-06-04)

### Docs

- **core**: cleaning up documentation ([#04457fa](https://github.com/pol-sb/ATLAS/commit/04457fa83367a9ab57b8444dbc9c1618a845f044))
- **core**: add CONTRIBUTING.md ([#9d57e25](https://github.com/pol-sb/ATLAS/commit/9d57e251d3e206fc43e5ea300cec6f4442103719))
- **core**: add security policy ([#16b28d3](https://github.com/pol-sb/ATLAS/commit/16b28d331c7280897a655516565305887bd25272))
- **core**: fix typo in README.md ([#9cb30e6](https://github.com/pol-sb/ATLAS/commit/9cb30e61fcd146e618e6be7afcb8a2e377466260))
- **core**: add images to README.md ([#da0f144](https://github.com/pol-sb/ATLAS/commit/da0f14424ee704655e4bfc61f85f2dc23e60f40b))
- **core**: update readme ([#9e5e922](https://github.com/pol-sb/ATLAS/commit/9e5e92225753009340cd0720fafcc0e5d59a1120))
- **core**: update readme ([#6b09ada](https://github.com/pol-sb/ATLAS/commit/6b09ada86fa0dbe3611fc00f135ae45df749bd73))

### Fix

- **core**: correct paths for versioning and documentation generation ([#532ee30](https://github.com/pol-sb/ATLAS/commit/532ee30880fb0e81dd46316556adb0a9c55f0bd7))
- **core**: display info panel during setup script and hide password entry ([#adb9fe4](https://github.com/pol-sb/ATLAS/commit/adb9fe48698353a3384b9c076527cf99dab7d049))

### Refactor

- **core**: missing atlas rebrand ([#2de61cf](https://github.com/pol-sb/ATLAS/commit/2de61cf2756b4ad5558a80b92a6c8f70c4e6dc3a))

### Style

- **core**: docstrings and comments formatting ([#c99ebad](https://github.com/pol-sb/ATLAS/commit/c99ebad52e9b82d35ea2a9aaf5af3f8358401371))
- **core**: change import names ([#5f6b427](https://github.com/pol-sb/ATLAS/commit/5f6b427274d8f4bc0c410afcac5b41b01c2200fb))

## 0.52.3 (2026-06-02)

### Docs

- **core**: update readme ([#f60e526](https://github.com/pol-sb/ATLAS/commit/f60e5265fa0d98807f3e45d274ec963bb7a166b2))
- **core**: center logos ([#4d519a7](https://github.com/pol-sb/ATLAS/commit/4d519a7a8c8cfe9477bb2bef626d4997a8e88719))
- **core**: add new logos v1 ([#afdb86f](https://github.com/pol-sb/ATLAS/commit/afdb86f1de442b5f51b1979dba5f53b8c1f65688))
- **core**: remove old logos ([#450589a](https://github.com/pol-sb/ATLAS/commit/450589aca861b65c3274f70fd6906525298e9db2))
- **core**: commit test on org repo ([#e1ab304](https://github.com/pol-sb/ATLAS/commit/e1ab304253f6f2b7ecc3a69aa2fa52bbed4977a3))
- **core**: upload dark version of AL picture to media ([#bc7a6ca](https://github.com/pol-sb/ATLAS/commit/bc7a6cac913f8d195ea7e83c5c1b29b2385ec086))
- **core**: update README.md ([#ba441c6](https://github.com/pol-sb/ATLAS/commit/ba441c61351c1c14dbdadb13b023e69605298ca3))

### Test

- **core**: added missing test suite ([#02326ad](https://github.com/pol-sb/ATLAS/commit/02326ad8369bb500df30976a9797831bc6ee9f9e))

## 0.52.2 (2026-06-01)

### Docs

- **core**: update citation file ([#74bc2af](https://github.com/pol-sb/ATLAS/commit/74bc2aff763f7ce6e0193725ad648d5856443f15))

### Refactor

- **core**: rebranding continuation ([#e6add9c](https://github.com/pol-sb/ATLAS/commit/e6add9cd8fd28dfadc570e3857aff91a942322f6))
- **core**: rebranding continuation ([#008196c](https://github.com/pol-sb/ATLAS/commit/008196ce03371c0eedc1e5b7dbd3654126612b77))

## 0.52.1 (2026-05-31)

### Docs

- **al_loop**: add missing scheduler options to test_db schema ([#d44bcd2](https://github.com/pol-sb/ATLAS/commit/d44bcd2c02e917d537ed6a359a2924f469cc4cc0))

### Misc

- **core**: update github documentation workflow ([#acf1814](https://github.com/pol-sb/ATLAS/commit/acf18148bb778dfcfac84935fe11d3db72c47f6f))

### Refactor

- **core**: ATLAS rebrand continuation ([#c906531](https://github.com/pol-sb/ATLAS/commit/c9065319004657ceea403a38705f4272e8778431))

## 0.52.0 (2026-05-31)

### Docs

- **core**: update docs ([#f836db1](https://github.com/pol-sb/MatDBForge/commit/f836db1ccdf25dc2273894aaaff8f64364ba30b6))

### Feature

- **core**: first round of renaming MatDBForge -> ATLAS ([#317d82d](https://github.com/pol-sb/MatDBForge/commit/317d82d62eee00d35f87206ab4bb3ef9c0d32717))
- **al_loop**: allow saving test set ([#a1e51b1](https://github.com/pol-sb/MatDBForge/commit/a1e51b151603a2a7d8a9c3a6030b85df92e6d09c))
- **init_db**: add util functions ([#2834c99](https://github.com/pol-sb/MatDBForge/commit/2834c99558497c12894e5ab1bcc30fd598edd3c8))

### Fix

- **core**: improve version check ([#49d7316](https://github.com/pol-sb/MatDBForge/commit/49d731682612c7314216ce11a48ea2a1da678b1c))
- **benchmarks**: allow several surfaces in run_defect_formation_energy_benchmark ([#969a54a](https://github.com/pol-sb/MatDBForge/commit/969a54a7fcdf381c50a0a8d0b26c6ffab70bb8f8))
- **md**: allow processing large structures in safeguard ([#5a3c2b6](https://github.com/pol-sb/MatDBForge/commit/5a3c2b6a76df8c1c63319e3f498601b6962d26d4))
- **core**: update report util to last al version ([#ca127c2](https://github.com/pol-sb/MatDBForge/commit/ca127c26f2d0c4a336aa10be0f5dfa8bb5a656f7))
- **domain_validity**: fix autoencoder description mapping ([#f0f876e](https://github.com/pol-sb/MatDBForge/commit/f0f876eabd4cdd9a5dffa4781225d75226e2a784))

### Refactor

- misc fixed ([#bb634ba](https://github.com/pol-sb/MatDBForge/commit/bb634ba72d07c44ce4ffdc174fe2806e4a02aca4))
- **domain_validity**: add save_descriptors() function ([#662c1c5](https://github.com/pol-sb/MatDBForge/commit/662c1c58241f7d5763983b9596095b47c4a160d9))

## 0.51.1 (2026-05-06)

### Fix

- **core**: remove repeated refresh_per_second ([#e64f5c0](https://github.com/pol-sb/ATLAS/commit/e64f5c03e196d3d04fc819888044388737aa751b))
- **core**: clear handlers in mdb logger when loading aiida profile to avoid duplicate log messages ([#04b71b6](https://github.com/pol-sb/ATLAS/commit/04b71b64b1848329ef1bbb4e927fd08271f6b1a5))

### Refactor

- **core**: reduce refresh_per_second in atl_show_progress() ([#5b9bc75](https://github.com/pol-sb/ATLAS/commit/5b9bc75025b4615bc96e9293ad7cb149dd96c640))

## 0.51.0 (2026-05-06)

### Feature

- **init_db**: add database analysis script ([#44ff63b](https://github.com/pol-sb/ATLAS/commit/44ff63b1ce9a0d66a8ad088703578f932008e72d))

### Fix

- **core**: reduce refresh rate of progress bar ([#8a5ef9e](https://github.com/pol-sb/ATLAS/commit/8a5ef9e1f8874e3f7c44815104d8cb9d96d0f9f8))

### Refactor

- **al_loop**: improve clarity of seed database loading when resuming active learning ([#feaf7c8](https://github.com/pol-sb/ATLAS/commit/feaf7c81e61576d97538805763fb70eb2953f41f))

## 0.50.2 (2026-05-03)

### Fix

- **benchmarks**: add floor to size in index selection for Vendi score subsampling ([#03ef705](https://github.com/pol-sb/ATLAS/commit/03ef70569d1fe4bc0fa9be2903174a40e34d3618))
- **domain_validity**: improve performance of get_latent_space_autoencoder() ([#661a57e](https://github.com/pol-sb/ATLAS/commit/661a57edc7803d30f780a56e02fa45eb70ac0e1d))
- **core**: reduce verbosity and logging duplicates ([#9860c13](https://github.com/pol-sb/ATLAS/commit/9860c13789843e64aaa584fe53c3e1185eb22f53))

### Performance

- **md**: improve MD safeguard filters ([#75ef7cc](https://github.com/pol-sb/ATLAS/commit/75ef7cc4f89392175a870ceec4b8aea0ba55ab19))

## 0.50.1 (2026-04-27)

### Fix

- **al_loop**: free up memory after FPS descriptor gathering to prevent memory leak ([#5b59c47](https://github.com/pol-sb/ATLAS/commit/5b59c47315fc8ae051130847cc5e1fcb89fad714))
- **core**: add missing scikit-image ([#9fbfb63](https://github.com/pol-sb/ATLAS/commit/9fbfb637059b7be2fc74e2fb569e52643ca08726))

### Refactor

- **domain_validity**: reduce logging verbosity for autoencoder ([#8958645](https://github.com/pol-sb/ATLAS/commit/895864518a47f063bcc3ecedc897202e1a5b7314))

## 0.50.0 (2026-04-20)

### Docs

- **schema**: reformat schema ([#26ee461](https://github.com/pol-sb/ATLAS/commit/26ee461295f35d2ba6c15cf28625007eb0395dcd))

### Feature

- **domain_validity**: add functions for morphological closing ([#0d220c2](https://github.com/pol-sb/ATLAS/commit/0d220c2968c9534ba61586c5159bd0ab37d9b01c))

### Fix

- **benchmarks**: improve performance and accuracy of diversity metrics ([#8ac6053](https://github.com/pol-sb/ATLAS/commit/8ac60534b8e37499b769491b2bcdf1d23e1d9173))
- **core**: compare FPS with correct structures every step ([#ea4ca0a](https://github.com/pol-sb/ATLAS/commit/ea4ca0ab0b14c13c71ba88bc33ee016ee7152f4f))
- **core**: misc fixes ([#9ff9788](https://github.com/pol-sb/ATLAS/commit/9ff97884c27d19092e0bfdf45933441c24e6f3eb))
- **core**: avoid use get instead of pop to get descriptor settings ([#89bf678](https://github.com/pol-sb/ATLAS/commit/89bf6782b3b44873a466cfdd7d36e12de2792712))
- **core**: update cuequivariance versions to avoid incompatibilities ([#c75eda6](https://github.com/pol-sb/ATLAS/commit/c75eda611b2be8910f95269319ec0c9aaa1f2596))

### Refactor

- **core**: reduce verbosity of module imports ([#530ccd9](https://github.com/pol-sb/ATLAS/commit/530ccd9f664b7dbf5855ec9d7c39577201fba2fb))
- **core**: add progress bar compatible with rich text logs ([#7e05b70](https://github.com/pol-sb/ATLAS/commit/7e05b7080d5ea4fe4322d4e90931d1d9ca16bda9))
- **core**: reduce verbosity when importing certain libraries ([#9380048](https://github.com/pol-sb/ATLAS/commit/938004854ef9c1f3ec5549af844b7239fed10042))
- **domain_validity**: improve clarity of latent space area normalization ([#17fd530](https://github.com/pol-sb/ATLAS/commit/17fd53045ddcc44e5da0ce8a0b8a4bcd64195873))

## 0.49.3 (2026-04-10)

### Fix

- **domain_validity**: add small scaling after hull optimization ([#a73b8ed](https://github.com/pol-sb/ATLAS/commit/a73b8ed3c8ecd8fd831647c90195840ea349918b))
- **domain_validity**: add standardization to autoencoder evaluation ([#bf32d55](https://github.com/pol-sb/ATLAS/commit/bf32d554bda37117cc92f72d0a3e0df333bc3901))
- **domain_validity**: pass missing bottleneck_dim parameter to autoencoder ([#925f6e1](https://github.com/pol-sb/ATLAS/commit/925f6e1cb6ce36557a5b867a2c95e9f67c973afa))
- **domain_validity**: add missing data standardization to autoencoder ([#6e0e4d2](https://github.com/pol-sb/ATLAS/commit/6e0e4d262760123e53c9f86bb43e05a3a0bdebe9))
- **domain_validity**: unclutter legend and higher res exports in quadtree/alpha-shape plots ([#777d593](https://github.com/pol-sb/ATLAS/commit/777d593eb5cd322e5a6a9bc3ac9f4cc6a0e8594f))
- **domain_validity**: disallow adding points to already subdivided leaves in quadtree ([#47ba1f5](https://github.com/pol-sb/ATLAS/commit/47ba1f56246042de0f8ec85ee9616b8c7dd738e4))
- **domain_validity**: include small epsilon in cost when minimizing area in alpha minimizer ([#a34e178](https://github.com/pol-sb/ATLAS/commit/a34e178a1a876e40ba5027f690b17f08408c0deb))
- **domain_validity**: Include alpha-based gradient in objective function to negative alpha penalty term ([#099ad81](https://github.com/pol-sb/ATLAS/commit/099ad8100e6f3b86af3df723e6d5532ffca4e17e))

### Refactor

- **domain_validity**: move locate_standarization_files to autoencoder.py ([#036efa7](https://github.com/pol-sb/ATLAS/commit/036efa7f4f76096c7ca76e63095fa526e0e62cee))

## 0.49.2 (2026-04-07)

### Docs

- **al_loop**: add missing scheduler options to test_db schema ([#86926ed](https://github.com/pol-sb/ATLAS/commit/86926ed9b2af7559fe8b7152acb82a5f2e237e0b))

### Fix

- **domain_validity**: allow autoencoder training to run when wandb=False ([#2fa7cfa](https://github.com/pol-sb/ATLAS/commit/2fa7cfa22425814b43bfafab2628b068796139fb))
- **domain_validity**: allow alpha shape function to continue when hulls with two points are found ([#4b4c0da](https://github.com/pol-sb/ATLAS/commit/4b4c0da487fbc8e5b727ffcba3999a295004e4d4))

## 0.49.1 (2026-04-02)

### Docs

- **domain_validity**: increase default alpha-shape alpha value range ([#975aa65](https://github.com/pol-sb/ATLAS/commit/975aa653cdb7307fafd7af9e8c78b20a5382fec7))
- **al_loop**: improve resume python argparse help entry ([#92cc24f](https://github.com/pol-sb/ATLAS/commit/92cc24f8e185f8ef2a1466ca081400b571fe7d99))

### Fix

- **al_loop**: enable usage of ImagePNGData class throughout active learning ([#fabaa5c](https://github.com/pol-sb/ATLAS/commit/fabaa5cbb25dc438e57092563af77e20ca5e703d))
- **domain_validity**: check autoencoder train data from train_data dataloader tensor ([#61d947a](https://github.com/pol-sb/ATLAS/commit/61d947a8da77743ee2b562090337020a9531e04b))
- **safeguard**: rename uuid variable overwriting uuid module in extrapolation scope ([#7fd10b3](https://github.com/pol-sb/ATLAS/commit/7fd10b37bbb2ce7f7ddd69853fbb17fd73ca633f))
- **benchmarks**: improve several benchmark tests from MLIP benchmark suite ([#93f0451](https://github.com/pol-sb/ATLAS/commit/93f04514ccb239476ad0d1c8de5e0518d8ac1b9b))
- **core**: add options for cli config generation for latent_space_analysis ([#3992331](https://github.com/pol-sb/ATLAS/commit/3992331037ae161dc21674782e2d6b85add14ac1))
- **domain_validity**: update inputs for generate_descriptors function in diversity_metrics.py ([#5148d6d](https://github.com/pol-sb/ATLAS/commit/5148d6d7770e168a6622413debf0c54fb085a550))
- **report**: catch TypeError when path is missing, add default ([#7f90fbf](https://github.com/pol-sb/ATLAS/commit/7f90fbf1121aca675cd1da11d7274ca15596cd61))
- **domain_validity**: add missing verbosity parameters to descriptor gathering functions ([#b7b00e3](https://github.com/pol-sb/ATLAS/commit/b7b00e38eb13cf6fee9c8a1ad7b6175a609aafb6))
- **domain_validity**: add alpha term to cost function on cases above threshold in alpha-shape optimizer objective function ([#ae9ac9e](https://github.com/pol-sb/ATLAS/commit/ae9ac9e1d8f35748ee12574178eebc2a6b20c9f4))
- **md**: change elif with if for the extrap_type != 'none' check ([#8726710](https://github.com/pol-sb/ATLAS/commit/87267101e135837cb25e98e7922b77677f41609d))
- **al_loop**: Add missing repeat option for base structures in safeguard ([#6d867b4](https://github.com/pol-sb/ATLAS/commit/6d867b4b7658fa4ecfbdd7088e935a62151b1ca9))
- **domain_validity**: improved verbosity settings when gathering descriptors ([#3e0996b](https://github.com/pol-sb/ATLAS/commit/3e0996b3c6bcd599c4b2c6936a1e78986293dfd8))

### Misc

- **core**: add exclude-newer for tool.uv ([#55b5303](https://github.com/pol-sb/ATLAS/commit/55b53036c84c510c7a1a71a07602fa3e99c0c034))

### Performance

- **domain_validity**: improve Autoencoder dataset splitting and data loading ([#a8d19c6](https://github.com/pol-sb/ATLAS/commit/a8d19c655aedb0d0a258676b1d7d44628d802150))

### Refactor

- **domain_validity**: add convenience methods to rectangle class ([#3924638](https://github.com/pol-sb/ATLAS/commit/3924638b97ab0a54097c908aa82a6690ac1d7c1d))

## 0.49.0 (2026-02-09)

### Feature

- **domain_validity**: add 'atl_latent_space_analysis' tool ([#906c630](https://github.com/pol-sb/ATLAS/commit/906c6309cd1a7e1eb9065272bee87b1171f35918))

### Fix

- **domain_validity**: fix check_traj_in_domain() implementation in concave_hull.py ([#662ccb4](https://github.com/pol-sb/ATLAS/commit/662ccb4c5dfbcd9d412dc72b55efc87367464b55))
- **al_loop**: add condition for missing stop_al_loop_error at end of loop ([#e0c570b](https://github.com/pol-sb/ATLAS/commit/e0c570beb854588ab29dc78c6288326b66946409))
- **core**: misc fixes ([#2e7571c](https://github.com/pol-sb/ATLAS/commit/2e7571cc85b5298b29e458ac0c282e0e52c4b7ca))
- **md**: add missing argument when unpacking atl_al_ut.generate_descriptors_soap ([#a558078](https://github.com/pol-sb/ATLAS/commit/a5580789520b8ed1c2635d0cc09bc95e1cd7bac4))
- **core**: return errors when failing to load toml file ([#3d34207](https://github.com/pol-sb/ATLAS/commit/3d3420794e6f5752c55872a8a816dfdaefcc738e))
- **core**: use lowercase in logger styles ([#9346719](https://github.com/pol-sb/ATLAS/commit/93467194f15ddcfcdb68aaa3a6e361a0776bca73))
- **al_loop**: allow containerized code usage for mlip train ([#3425d88](https://github.com/pol-sb/ATLAS/commit/3425d886a22b3147e77bf10d51c5be3628e6810d))

### Misc

- **core**: update LICENSE ([#ca3cbc1](https://github.com/pol-sb/ATLAS/commit/ca3cbc172a2a71f7859ca4ccf7e298fb8d412a4b))

### Refactor

- **domain_validity**: move check_traj_in_domain() to concave_hull.py ([#20aa47d](https://github.com/pol-sb/ATLAS/commit/20aa47d47d524546bb4d179b65b6bb781c34ad6d))

## 0.48.3 (2026-02-04)

### Fix

- **domain_validity**: index the only entry on concave_hull list when creating Polygon ([#1f3ba90](https://github.com/pol-sb/ATLAS/commit/1f3ba902cb7d75fedddc88f1debfaa9df96f7279))

## 0.48.2 (2026-02-04)

### Fix

- **domain_validity**: instantiate missing concave hull variable ([#5f96c9e](https://github.com/pol-sb/ATLAS/commit/5f96c9eda3c792bdcba4ab65b57dcb1d27e90d10))
- **al_loop**: add config and uuid check ([#d64709c](https://github.com/pol-sb/ATLAS/commit/d64709c59c6aa6383d55bf9e7ac3c36baed3e407))

## 0.48.1 (2026-02-03)

### Fix

- **core**: improve log formatting and avoid logger overlap ([#83a2e4c](https://github.com/pol-sb/ATLAS/commit/83a2e4cbbdff10ffc69a675a153b963fe47eca0e))
- **core**: add uuid to log filename to ensure file uniqueness ([#275ab11](https://github.com/pol-sb/ATLAS/commit/275ab110dbeb7e8b0ba608bb623a44cf5b1ac1b1))

### Misc

- **core**: bump aiida from 2.7.1 to 2.7.3 ([#89bc17d](https://github.com/pol-sb/ATLAS/commit/89bc17df79d21006d45f9b49273efb5135d949b0))

## 0.48.0 (2026-01-30)

### Feature

- **al_loop**: introduced quadtree+optimizer concave hull. ([#2967664](https://github.com/pol-sb/ATLAS/commit/2967664282128fefd52f2f8ed6abe3616ed98a02))

### Fix

- **core**: update usage of generate_descriptors() ([#f8a73f9](https://github.com/pol-sb/ATLAS/commit/f8a73f952570e6ce4cf2ad0e474f903fe0a3a162))
- **al_loop**: return uuid_list when getting soap descriptors ([#289ceb1](https://github.com/pol-sb/ATLAS/commit/289ceb127c951798a8a38382a7ba99a390023568))

## 0.47.3 (2026-01-26)

### Docs

- **core**: dump toml section information on each section when automatically generating input files ([#c7ab27f](https://github.com/pol-sb/ATLAS/commit/c7ab27fda6e09a302b5dea7168df0c40de1f86c0))
- **benchmarks**: improve documentation for benchmarks ([#3fe1617](https://github.com/pol-sb/ATLAS/commit/3fe161781ade45346b5ea34730d7d9bb92e93caf))

### Fix

- **benchmarks**: fix defect energy and surface energy benchmarks ([#1ecb42e](https://github.com/pol-sb/ATLAS/commit/1ecb42e53510bffe33bdbf3631a9dae1c61c1ef0))

### Misc

- **benchmarks**: add input file validation to mlip benchmark ([#a8da225](https://github.com/pol-sb/ATLAS/commit/a8da225dd83af8d28cd4cb649fc163f79ab8c1f4))
- **core**: limit torch version below 2.10, force plotly>=6.1.1 ([#6a1f0e2](https://github.com/pol-sb/ATLAS/commit/6a1f0e2f9cdf4df51bcb6038f82d46b4ec143141))

### Refactor

- **core**: add types and keyword arguments to logging functions ([#7860b04](https://github.com/pol-sb/ATLAS/commit/7860b045e95b2dea4aadff59ae7e058bc66991d6))

### Style

- **core**: correct mistake in error message during input validation ([#2ecfaf2](https://github.com/pol-sb/ATLAS/commit/2ecfaf2caf2a022600b74effe84030a0f1bc026b))
- **md**: add extrapolation and md performance prints, update md script with easier to understand variables ([#c438998](https://github.com/pol-sb/ATLAS/commit/c438998a41df45d9b3f22b3e1914221da94268bf))

## 0.47.2 (2026-01-19)

### Fix

- **core**: allow generation of mlip_benchmark toml input file using atl_gen_configuration_file ([#f7a7781](https://github.com/pol-sb/ATLAS/commit/f7a7781ef5ddda5582fa962d3f5c6b5293dbccc9))

## 0.47.1 (2026-01-14)

### Docs

- **schema**: improve schema and properly check for missing keys. ([#b7449aa](https://github.com/pol-sb/ATLAS/commit/b7449aab95bf4eae5a74eac98d72f43b4584b0f2))

### Style

- **report**: reformat report_utils.py ([#fcbda65](https://github.com/pol-sb/ATLAS/commit/fcbda6546401f16be132a8476f7045279653239b))

## 0.47.0 (2026-01-14)

### Docs

- **al_loop**: fix docstring of add_single_atoms() ([#99b75a5](https://github.com/pol-sb/ATLAS/commit/99b75a5411c814752e61cd15abf742770781706e))
- **schema**: correctly arrange md.params options ([#cadacaa](https://github.com/pol-sb/ATLAS/commit/cadacaa78efde7c6dd1399114d84c01b416799f0))
- **core**: allow description in dynamic_keys sections and add `schema_under_dynamic_keys` ([#2971237](https://github.com/pol-sb/ATLAS/commit/29712376ec3af7b2467e1ca3b6fa5e5f775fcafc))

### Feature

- **init_db**: implement RBF Vendi score and optimize metrics ([#d3402fa](https://github.com/pol-sb/ATLAS/commit/d3402fabf63d47e46325c0e06fd31e8543d1d07e))
- **init_db**: enhance InitialDatabase database loading and ASE compatibility ([#c88c2c0](https://github.com/pol-sb/ATLAS/commit/c88c2c09ea4ffe9675107dd5ca10a4d9e26a763c))
- **core**: add MLIPModelFileData and ImagePNGData custom datatypes ([#ba7fe44](https://github.com/pol-sb/ATLAS/commit/ba7fe44a93d7b58a11c20b49e9c65dde632bcf11))

### Fix

- **core**: correct CLI config validation and parameter parsing ([#62ffbce](https://github.com/pol-sb/ATLAS/commit/62ffbce7b81152345e4c7127b091a4e87ba61764))
- **safeguard**: improve input parameter handling and file cleanup ([#0e5bb3f](https://github.com/pol-sb/ATLAS/commit/0e5bb3fb28087bc39811611754b51b240b027b1c))
- **al_loop**: add CUDA checks and fix workflow logic ([#32022a2](https://github.com/pol-sb/ATLAS/commit/32022a22c0bc8b2c0ad8396947ea936e52411bbe))
- **core**: add timeout to git-remote version check ([#9ac22f6](https://github.com/pol-sb/ATLAS/commit/9ac22f6f7feaf8bb4da6ba26a5dad120e76d2809))
- **al_loop**: properly update dft_calc_list ([#8627b4b](https://github.com/pol-sb/ATLAS/commit/8627b4b5eedd77318410f07e62e7d21e4d067c78))
- **al_loop**: convert dict containing serialized Atoms object to Atoms in `sampler_populate_E_and_F_list()` before predicting E and F ([#2428531](https://github.com/pol-sb/ATLAS/commit/2428531dd97b54fed9ddb61cc5cd23c51b8375f7))
- **al_loop**: rename function all argument for sampler_populate_E_and_F_list() ([#fce473a](https://github.com/pol-sb/ATLAS/commit/fce473a1037b2e1bec76618f684dd2313abc4ef9))
- **al_loop**: allow data acquisition filter to proceed when DFT calculations are used ([#5e6252a](https://github.com/pol-sb/ATLAS/commit/5e6252a9c44a1672b2bc8b3154f5548511a2be05))
- **dft**: prevent dft calculations from running when atl_ids are not unique ([#256568e](https://github.com/pol-sb/ATLAS/commit/256568ea7c50fb8be47280c771515e3c472d8bc8))
- **al_loop**: prevent al_loop from running when repeated uuids are present ([#f2c4278](https://github.com/pol-sb/ATLAS/commit/f2c42788d4f5253f027f94db3d4f0a687cd3a013))
- **schema**: remove extra '' from default entries ([#2308215](https://github.com/pol-sb/ATLAS/commit/23082150380126a892c9c3ac43d1c386f7f4024c))
- **md**: add missing `use_for_structure_types` check for stages ([#2ae8b3c](https://github.com/pol-sb/ATLAS/commit/2ae8b3cc1b3ecfbc2ced692b63a0456213b508c7))
- **al_loop**: change incorrect usage of aiida_uuid ([#3ff8d1a](https://github.com/pol-sb/ATLAS/commit/3ff8d1a478775935e69cb0ee3bccb7ab4a9d1e6c))
- **al_loop**: replace named files with wildcard in `retrieve_temporary_list` of `GetDescriptorsCombinedCalculation` ([#1367fb3](https://github.com/pol-sb/ATLAS/commit/1367fb32ebea6188244386e26795a162d25ee08f))
- **al_loop**: allow running AL Loop when not enough small structures ([#6b49a80](https://github.com/pol-sb/ATLAS/commit/6b49a8049c5e3eed9ce3ee6ad50c4126bd45ec5b))
- **al_loop**: properly copy eval information to test db eval calcjob by using open() with tmpfile ([#94434d8](https://github.com/pol-sb/ATLAS/commit/94434d83553df0d8834b423abda845cb827a119c))

### Misc

- **core**: update dependencies and fix ruff settings ([#73646f4](https://github.com/pol-sb/ATLAS/commit/73646f44ede57cfaf8b4dacc8351a37734542c63))

### Refactor

- **al_loop**: update print messages for al_loop dft filter ([#5fd2f93](https://github.com/pol-sb/ATLAS/commit/5fd2f93ca7ae5b15c40a6e5673040dfe55afa395))
- **al_loop**: format code and make errors verbose ([#6c24530](https://github.com/pol-sb/ATLAS/commit/6c2453043ecb99807d3c6482c75c8e9d78538608))
- **al_loop**: change units from eV to meV under `eval_test_db.py` ([#313e82b](https://github.com/pol-sb/ATLAS/commit/313e82be4474df9188415ee907218e50f55c2efc))
- **al_loop**: clean unused code ([#f74291f](https://github.com/pol-sb/ATLAS/commit/f74291f687fd28661edd89e82de9c7e563157483))
- **domain_validity**: re-add logger to `atl_check_descr_combined.py` ([#32a0d67](https://github.com/pol-sb/ATLAS/commit/32a0d672afd61673b09bc59f7ca6a56c8204ef9f))

### Style

- **core**: update type hints and formatting ([#d4ff4c1](https://github.com/pol-sb/ATLAS/commit/d4ff4c169f60957831e5d2a67e44fb487637fbef))

## 0.46.9 (2025-12-02)

### Fix

- **al_loop**: add new error code for failed descriptors and use it to stop al loop when descriptors fail ([#19d28b1](https://github.com/pol-sb/ATLAS/commit/19d28b167473d871537a4f2d25ae789bfaf26578))
- **al_loop**: update `test_db_eval_results` dict instead of overwriting it every iteration ([#6a91ea5](https://github.com/pol-sb/ATLAS/commit/6a91ea52397de7fa46ba79dae00378bdbbcbbb15))
- **domain_validity**: properly gather descriptors from npz file ([#6e47981](https://github.com/pol-sb/ATLAS/commit/6e4798114de2fb8d91576231eb857dd7c32a109d))
- **domain_validity**: check against loaded file when defining `point_arr` in `train_autoencoder.py` ([#c27c977](https://github.com/pol-sb/ATLAS/commit/c27c977d02c712387a031e84e08969ebdce6be28))
- **al_loop**: FINALLY fix infinite safeguard loop ([#eef099f](https://github.com/pol-sb/ATLAS/commit/eef099f7cc38437bf5a39b8ee51560d4a138f14c))

### Refactor

- **al_loop**: rearrange prints and function calls ([#694071c](https://github.com/pol-sb/ATLAS/commit/694071c23c4c2cacb403ef83248c27d7cfa2654e))

## 0.46.8 (2025-12-01)

### Fix

- **al_loop**: properly catch "empty" strings when given as default arguments for descriptor calc ([#6f4c63e](https://github.com/pol-sb/ATLAS/commit/6f4c63ed2f72bfaf0d723bdb98a03de8fef69cd7))

### Refactor

- **al_loop**: remove unused prints ([#c6956cf](https://github.com/pol-sb/ATLAS/commit/c6956cf16f165150c38d0fce2dd0a50029fe9e29))

## 0.46.7 (2025-12-01)

### Fix

- **al_loop**: catch empty strings for the default parameters in the "descriptors" stage ([#62db350](https://github.com/pol-sb/ATLAS/commit/62db3503d1a4f7c28c264b3ab9e7d2262612f04d))

## 0.46.6 (2025-12-01)

### Fix

- **al_loop**: initialize error tracking variable to False at the start of the ALStep ([#96b4017](https://github.com/pol-sb/ATLAS/commit/96b4017c0e7485240167f7dd9ce26353625f9dcf))

### Refactor

- **al_loop**: improve clarity of log messages ([#636142d](https://github.com/pol-sb/ATLAS/commit/636142d18c7c4432cb3b51f2464ebfb3daf3262c))

## 0.46.5 (2025-12-01)

### Fix

- **al_loop**: properly parse string-like None types in `apply_defaults()` ([#71fffba](https://github.com/pol-sb/ATLAS/commit/71fffba6f081b9c77883177cea8c4e9474468506))
- **al_loop**: make stop_al_loop_error an output of `SimpleActiveLearningWorkChain` to send it to Base context ([#14715be](https://github.com/pol-sb/ATLAS/commit/14715be215c67bc8fd825e214791b70787ce38e5))

## 0.46.4 (2025-11-30)

### Docs

- **core**: improve report for warnings ([#93b202c](https://github.com/pol-sb/ATLAS/commit/93b202cee117b17f1b34eca8dcedfe8130c841bd))
- **core**: assign wildcard entries to proper section when generating docs ([#0842655](https://github.com/pol-sb/ATLAS/commit/08426559ca03e9ea2d62c1c6a500fa57113e49f9))

### Fix

- **al_loop**: add missing definition of self.ctx.test_settings when test_db is disabled to allow AL to run ([#df9c141](https://github.com/pol-sb/ATLAS/commit/df9c14105b3e85ea9fe80c1972e9abc5d42f56fd))

## 0.46.3 (2025-11-28)

### Docs

- **al_loop**: improve documentation for test database check ([#92d972b](https://github.com/pol-sb/ATLAS/commit/92d972b4d3a13e81b64f17d46f35d3c1d6a5778f))

## 0.46.2 (2025-11-28)

### Docs

- **core**: Add missing dependency keys. Make `safeguard` section mandatory to force users to either enable it or disable it ([#0fbfe37](https://github.com/pol-sb/ATLAS/commit/0fbfe37659f54e3abb697ee8cd0e415ab7292df1))

### Fix

- **al_loop**: ignore safeguard check if safeguard is disabled ([#9667cf2](https://github.com/pol-sb/ATLAS/commit/9667cf2f574a11fcdd344328cd02780878bb3be9))
- **al_loop**: create copy of initial training database and save test set to base workchain ([#f98bc39](https://github.com/pol-sb/ATLAS/commit/f98bc39a64ad18a423b219d77ef5b68f4f2109ce))
- **al_loop**: add "calc_performed" tag to all new DFT structures during AL run ([#d9042ec](https://github.com/pol-sb/ATLAS/commit/d9042ec19bd3c3aefb41c8055a18e9adcb6930c9))
- **dft**: improve "stress" parsing from VASP ([#0629f75](https://github.com/pol-sb/ATLAS/commit/0629f756e8dd2813257b8625dc5deddd49370dc0))
- **al_loop**: add missing update to test db eval results if available ([#db36741](https://github.com/pol-sb/ATLAS/commit/db3674187e100edca7c8ba8fc7f5568381d4a37c))
- **al_loop**: avoid al_loop trying to resume from default `''` empty descriptor calc ([#d3f55a3](https://github.com/pol-sb/ATLAS/commit/d3f55a38d1bb473712b17521082578de31e61792))
- **core**: add missing variables in all return statements in `validate_config_file()` ([#7bb368b](https://github.com/pol-sb/ATLAS/commit/7bb368bb58d2bf372d8666394d25e342dcacd5d7))
- **core**: use `tomlkit` to write modified toml files to avoid format changes. add `tomlkit` dependency. ([#5522710](https://github.com/pol-sb/ATLAS/commit/552271046eac1d3d51a96ebb2f4873e5053ae7a7))
- **domain_validity**: changed default numpy array name from `arr_0` to `descriptor` ([#5176854](https://github.com/pol-sb/ATLAS/commit/51768542897e873d4a9fc13f92e811026866a4fc))

### Refactor

- **al_loop**: remove unused code ([#4ceb2ff](https://github.com/pol-sb/ATLAS/commit/4ceb2fffba5ee1403a861a8ca282ebcc22b6e048))
- **al_loop**: fix document and add TODO block for dynamic VASP DFT input loading ([#2987ba7](https://github.com/pol-sb/ATLAS/commit/2987ba7821d3d9ad777815e856caedf3168a6c41))
- **md**: changed `dim_red_method` check to switch case ([#2664d5e](https://github.com/pol-sb/ATLAS/commit/2664d5e0a45e12223161d2401b96489192d00488))

## 0.46.1 (2025-11-26)

### Docs

- **dft**: add information about ATL_DEFAULT in kspacing settings ([#2a80d62](https://github.com/pol-sb/ATLAS/commit/2a80d62564b9b1ee00f707718d68705a95d67582))

### Fix

- **al_loop**: enable warning usage in resume al calcs ([#e8a511a](https://github.com/pol-sb/ATLAS/commit/e8a511a19efb07b0a1946409c2b0eb3556618026))

## 0.46.0 (2025-11-26)

### Feature

- **al_loop**: add EvaluateMACEConfigsCalculation to active learning loop ([#929bfda](https://github.com/pol-sb/ATLAS/commit/929bfda0821330c9b4752324737a71f51b6d7067))
- **al_loop**: add toml check and warnings to al_loop start ([#3a53375](https://github.com/pol-sb/ATLAS/commit/3a533759b93f96ab7b676933db6f7aa5353850a8))
- **al_loop**: allow continuing with previous D_s when resuming an AL loop ([#46c6d50](https://github.com/pol-sb/ATLAS/commit/46c6d5088ee267d3d349fde94a5508e04d198050))

### Fix

- **dft**: assign default phase based on struct type if missing ([#72a45cb](https://github.com/pol-sb/ATLAS/commit/72a45cb25deef0aa4504ba08cbd06b55505cc2f6))
- **al_loop**: fix stress gathering ([#c6731b6](https://github.com/pol-sb/ATLAS/commit/c6731b69ebf35a7bbbe329f2d5b03714da839a15))
- **al_loop**: use rng seed in md and safeguard ([#cf2fe73](https://github.com/pol-sb/ATLAS/commit/cf2fe73bdf14934eb3decf4a9e44642bd87729c7))

### Misc

- **core**: add datashader dependency ([#884664f](https://github.com/pol-sb/ATLAS/commit/884664fd432e23c7719fb4c89df7624859e9c2cd))

### Refactor

- **al_loop**: add `return_code_from_settings()` to simplify container/portablecode usage ([#60c239d](https://github.com/pol-sb/ATLAS/commit/60c239d2914d31b0326debf767c050545a139ab9))
- **core**: output and formatting changes ([#f502921](https://github.com/pol-sb/ATLAS/commit/f502921cf1b85575f439a7021a03fc5561e2f031))

## 0.45.4 (2025-11-21)

### Fix

- **al_loop**: remove leftover debug code ([#b88080e](https://github.com/pol-sb/ATLAS/commit/b88080e934555e3513fe4357057ebc427f7c2141))

## 0.45.3 (2025-11-19)

### Docs

- **schema**: update schema ([#4633817](https://github.com/pol-sb/ATLAS/commit/463381731e78ac175127b7e694f9106bcfbd1e1e))

### Fix

- **core**: fix for docs workflow ([#4aab4ce](https://github.com/pol-sb/ATLAS/commit/4aab4ce3118eb41bd8b19a9272d27c0bbbcbe682))

### Misc

- **core**: fix workflow ([#980adbf](https://github.com/pol-sb/ATLAS/commit/980adbfda56ca2357a98599f628c6c35663a700f))

## 0.45.2 (2025-11-19)

### Misc

- **core**: fix docs conf and github workflow ([#d543dbd](https://github.com/pol-sb/ATLAS/commit/d543dbd115d916cb4743536aa83caa4c3f998b03))

## 0.45.1 (2025-11-19)

### Fix

- **core**: bump ase version to `3.26.0` or higher ([#d552771](https://github.com/pol-sb/ATLAS/commit/d5527718eef5d9c3b7dc052b6f290253f310f5c5))

### Misc

- **core**: remove cache from docs workflow and limit pages to build to last 10 versions ([#68dccce](https://github.com/pol-sb/ATLAS/commit/68dccce806388540cfbea07f7852586c04d6465e))

## 0.45.0 (2025-11-19)

### Docs

- **core**: allow usage of wildcard entries in input file schema for documentation generation ([#4828d82](https://github.com/pol-sb/ATLAS/commit/4828d82a83a2c71e86c57a62754d478b85d40eaf))
- update schema and improve tools.md ([#efab748](https://github.com/pol-sb/ATLAS/commit/efab74847132a84c805fd21a87b965dfa2b8463b))
- **core**: fix md schema and allow use of wildcards in schema ([#a048d61](https://github.com/pol-sb/ATLAS/commit/a048d61f538519d5d81869535058060048cae856))
- **md**: add inputs and docs for npt ensembles ([#0f3c5a8](https://github.com/pol-sb/ATLAS/commit/0f3c5a8db8a107bb6b0e0dc660358444e3912068))

### Feature

- **md**: allow usage of stages during md simulations ([#685b5e7](https://github.com/pol-sb/ATLAS/commit/685b5e7396098fd0ae5949f09be634101fb8a5e9))

### Fix

- **al_loop**: avoid missing ntfysh parameters during AL launch ([#77f9320](https://github.com/pol-sb/ATLAS/commit/77f93200f386f6a675f43206ea47d0609011bc09))
- **report**: improve report reliability ([#bc9c71d](https://github.com/pol-sb/ATLAS/commit/bc9c71dce16c335ac0a7cc3ce0a4c62bde1dda83))
- **al_loop**: fix stress key gathering, with voigt added ([#9593498](https://github.com/pol-sb/ATLAS/commit/9593498fde4d85cbf94bb3202fe9d85c3dff69f8))
- **al_loop**: add names to loggers so they are not initialized repeatedly, causing duplicated log messages ([#968faa0](https://github.com/pol-sb/ATLAS/commit/968faa0fe837fac0c58b9cb52d354bbd507ddae8))
- **md**: replace default melchionna-npt calculator with MTKNPT calculator ([#dba18e2](https://github.com/pol-sb/ATLAS/commit/dba18e2469d10c467953a0ac5cbb6b0c86e0e14f))

### Refactor

- **domain_validity**: add title parameter and extend documentation for `plot_concave_hull()` function ([#efba2f3](https://github.com/pol-sb/ATLAS/commit/efba2f30541a95e8adea6a5bc2fd4014689ced66))

### Style

- **domain_validity**: change alpha and area values to scientific notation ([#94646a7](https://github.com/pol-sb/ATLAS/commit/94646a727a6a59098ce591152d1b445c6387f002))

## 0.44.2 (2025-11-08)

### Docs

- **core**: change docs to improve key visibility ([#1a27742](https://github.com/pol-sb/ATLAS/commit/1a27742ef859417f976d09d74d9163ab0dbb6958))
- **core**: test new docs formatting for keys ([#dff7f44](https://github.com/pol-sb/ATLAS/commit/dff7f44ce4b9e55fffc4270d1e2c7a29d3395680))
- **core**: improve docs generation ([#7a27abf](https://github.com/pol-sb/ATLAS/commit/7a27abfef0c4668ea6be5869b90aff155288a2f8))
- **core**: add custom color for keys in documentation ([#accb882](https://github.com/pol-sb/ATLAS/commit/accb8825a7b5b9a03c7fb374e2691287f52881ff))

### Misc

- **core**: Add cleanup step to save space ([#1db3151](https://github.com/pol-sb/ATLAS/commit/1db31513416670861f3fd7e6e682b99e811832d9))

## 0.44.1 (2025-11-07)

### Fix

- **al_loop**: add dynamic input loading to descriptor calculation ([#9862953](https://github.com/pol-sb/ATLAS/commit/98629537c362ad808f10e71bc84e4d60d4cabd34))

### Refactor

- **al_loop**: enhance al_loop reporting and logging ([#33d991a](https://github.com/pol-sb/ATLAS/commit/33d991a704f8a65f4cc10b95de8f3f39b08e04c4))

## 0.44.0 (2025-11-07)

### Docs

- **md**: improve MD CalcJobs help messages ([#007684f](https://github.com/pol-sb/ATLAS/commit/007684f4cea917468f639eff7e98cde538f56bfe))

### Feature

- **al_loop**: add `load_md_calcs` AL input to load previously finished MD calculations ([#c72699d](https://github.com/pol-sb/ATLAS/commit/c72699d23b2facd23403ec94ec6ef3a0e7f7d130))
- **domain_validity**: allow mace foundation models for mace descriptors in descriptor calculation ([#b4f0817](https://github.com/pol-sb/ATLAS/commit/b4f08170b7b36d8098d8b7bef033a0d3c16a48df))
- **domain_validity**: add `outer_average` option to MACE descriptors ([#051fa4b](https://github.com/pol-sb/ATLAS/commit/051fa4bbb8762f02968a8b30f38d3e9221cbce61))

### Fix

- **al_loop**: change `current_settings` initialization to empty dict instead of None to avoid errors with the .get() method ([#be917df](https://github.com/pol-sb/ATLAS/commit/be917df1512593fa64849e24c7cc6481cb39cbf9))
- **al_loop**: ensure error state in `SimpleActiveLearningWorkChain` is properly tracked ([#4c46d3c](https://github.com/pol-sb/ATLAS/commit/4c46d3c0810d493488cce1d9d500cf14bb064749))
- **domain_validity**: make colorbar optional in `plot_concave_hull` ([#eb821af](https://github.com/pol-sb/ATLAS/commit/eb821af1324e952a5eecb7cf3511e75f6647cd8b))
- **domain_validity**: generate structure uuid on the fly during mace descriptor calculation ([#4a8ca9a](https://github.com/pol-sb/ATLAS/commit/4a8ca9a4acb34eed6cbfad96cc479bb590354586))

### Misc

- **core**: upgrade aiida-core version from `2.6.2` to `2.7.1` ([#a0480e9](https://github.com/pol-sb/ATLAS/commit/a0480e94f3392744ce8e7a3d6484f24a7f591fe1))

### Refactor

- **al_loop**: change `safeguard_not_attempted` to more robust `should_run_safeguard` check ([#08910c3](https://github.com/pol-sb/ATLAS/commit/08910c3a140779325a4fad091c70f333dc1a1b01))

### Style

- **al_loop**: add additional informative reports for MD calculation submission ([#36f2f9f](https://github.com/pol-sb/ATLAS/commit/36f2f9f636edf787780fb39858827342566a3551))

## 0.43.9 (2025-11-05)

### Fix

- **al_loop**: add `cls.safeguard_not_attempted` to al_loop to ensure proper execution of safeguard ([#bf137b9](https://github.com/pol-sb/ATLAS/commit/bf137b96e964a55722dcfee0818ec93a84a7a4ec))
- **al_loop**: improve check for existence of autoencoder model and concave hull file in md inputs ([#2cd6d7f](https://github.com/pol-sb/ATLAS/commit/2cd6d7f924fbd03a738bdbc872d24c315c82c3f1))
- **domain_validity**: remove extra commas from prints in `atl_check_descr_combined.py` script ([#c6a8656](https://github.com/pol-sb/ATLAS/commit/c6a86564608092659200d055b9f666edc5770f62))

## 0.43.8 (2025-11-04)

### Fix

- **safeguard**: improve check for `safeguard_errored_ids` ([#da743e7](https://github.com/pol-sb/ATLAS/commit/da743e7260bdc791de27b7921c136038755bd223))

### Misc

- **core**: pin cuequivariance versions to `0.5.1` to allow usage of old mace models ([#602fc73](https://github.com/pol-sb/ATLAS/commit/602fc733eecf6c4f3aea8a4611ba5d98a510d3c6))
- **safeguard**: add safeguard section to cz commit messages ([#87741fa](https://github.com/pol-sb/ATLAS/commit/87741fac26af59de41eef500079afb6a7207fc14))

## 0.43.7 (2025-11-03)

### Fix

- **al_loop**: check if `stop_al_loop_error` both exists and is true before stopping loop ([#d0ba190](https://github.com/pol-sb/ATLAS/commit/d0ba19013b3221adb9bc31ccbe92405a9021ef7c))

## 0.43.6 (2025-11-03)

### Fix

- **domain_validity**: add density to concave hull plot ([#f886869](https://github.com/pol-sb/ATLAS/commit/f8868698ba93ebe8c6087115c557c09ce87838ae))
- **al_loop**: use correct script for safeguard calcjob ([#553886c](https://github.com/pol-sb/ATLAS/commit/553886c2e73d9ad2bbf8ebb532ab400289261ac8))
- **domain_validity**: allow SOAP descriptor generation function to generate uuid4 on the fly ([#626b6ad](https://github.com/pol-sb/ATLAS/commit/626b6adbf19847a1532b16167e0e3480d94a3b0e))

### Misc

- **core**: pin version of `scipy` to `1.16.2` ([#3e94ed8](https://github.com/pol-sb/ATLAS/commit/3e94ed887ccbfbdf83d4c2a45410cd66750277ec))

### Refactor

- **al_loop**: improve logging for al_loop ([#85e26c6](https://github.com/pol-sb/ATLAS/commit/85e26c6260738c77cca6a987fc261c09a17dd0d0))

## 0.43.5 (2025-11-02)

### Fix

- **al_loop**: avoid infinite loop by not setting `self.ctx.stop_al_loop_error` to True and checking for errored safeguard calcs. ([#b1d9d26](https://github.com/pol-sb/ATLAS/commit/b1d9d26a4eee330f08069e406041c1ba8fa5f376))

## 0.43.4 (2025-10-31)

### Fix

- **al_loop**: check for existence of `concave_hull` array using hasattr ([#13a6cb7](https://github.com/pol-sb/ATLAS/commit/13a6cb7d848e2a07400812c8674d3750e8de1a15))
- **al_loop**: add mising `self.ctx.curr_md_all_structures_in_domain` definition ([#b4bb38f](https://github.com/pol-sb/ATLAS/commit/b4bb38fe647cc961b9fe51e0f78b40110ac9e3ef))

## 0.43.3 (2025-10-30)

### Docs

- **schema**: add commit explanation to doc generating utility ([#22f6818](https://github.com/pol-sb/ATLAS/commit/22f681819ac93508c68e05bbe3f84540d858ee36))
- **schema**: apply correct indentation for critical_notifications key ([#3295e7d](https://github.com/pol-sb/ATLAS/commit/3295e7d3c461654b93536b43251d359bf59f7f03))

### Fix

- **al_loop**: use `self.ctx.stop_al_loop_error` to determine safeguard stop ([#0c31d3e](https://github.com/pol-sb/ATLAS/commit/0c31d3ec7ac13f644eb474e4608713dba12547e8))

## 0.43.2 (2025-10-30)

### Docs

- **schema**: add 'name_pretty' fields to missing entries in schema ([#7f88526](https://github.com/pol-sb/ATLAS/commit/7f88526a8f4b8b1be2d79f6214f1a28ebaa296b2))

### Fix

- **schema**: use name_pretty field for documentation headers ([#5368b5a](https://github.com/pol-sb/ATLAS/commit/5368b5a49c0012e22d7272860511ed17645c6b52))

## 0.43.1 (2025-10-29)

### Fix

- **al_loop**: al_loop not proceeding when new safeguard is disabled ([#9764f99](https://github.com/pol-sb/ATLAS/commit/9764f9908158872d1e40456a339ee79231513512))

## 0.43.0 (2025-10-28)

### Docs

- **schema**: add explanation for benchmark tools ([#8155482](https://github.com/pol-sb/ATLAS/commit/815548235db26b010b5b73f2fb70c2f5fbadcb21))

### Feature

- **schema**: enable use of interpolation key ([#845435f](https://github.com/pol-sb/ATLAS/commit/845435f1f852062b2b9fc78774f4c13a85d7d7dd))
- **al_loop**: fix implementation of safeguard ([#7dadba1](https://github.com/pol-sb/ATLAS/commit/7dadba187068df4522ad58ee1952dbbb9dcd9039))

## 0.42.0 (2025-10-21)

### Docs

- **schema**: add 'mlip_benchmarks' section to schema ([#49a75d1](https://github.com/pol-sb/ATLAS/commit/49a75d1945a6eb59f3dd41abc67eda5220a4cf06))

### Feature

- **domain_validity**: improve alpha-shape alpha iterative selection ([#fa6e573](https://github.com/pol-sb/ATLAS/commit/fa6e5737a1742c70a72b5a5e33f66ce6cf55ee49))
- **benchmarks**: add toml input to benchmarks and finish implementation of coexistence benchmark ([#72920ff](https://github.com/pol-sb/ATLAS/commit/72920ff3c351cadeb001c67d57c1eb9093880e78))

### Fix

- **schema**: update parser for computer resources in toml checker ([#4507b0a](https://github.com/pol-sb/ATLAS/commit/4507b0aa26b0bd44109c1841bdda6199f36a2272))

## 0.41.4 (2025-10-10)

### Fix

- **al_loop**: allow disabling extrapolation and add extra options for advanced and basic extrapolation ([#9fb221b](https://github.com/pol-sb/ATLAS/commit/9fb221bae65ea1dcdcc549f691ac0e40abf50a37))

### Refactor

- **al_loop**: cleanup simple_active_learning.py and add notifications for all stages ([#1cae3db](https://github.com/pol-sb/ATLAS/commit/1cae3dbef834e814a38679ad56eaf1d900642c11))

## 0.41.3 (2025-10-10)

### Fix

- **al_loop**: make ntfysh_topic optional so it does not prevent AL from running if not passed in toml settings ([#dbe86eb](https://github.com/pol-sb/ATLAS/commit/dbe86eb9007d264ed20bcc1e5cbfd5c3f237558e))

### Misc

- **core**: bump up minimum scipy version from 1.14.1 to 1.16.1 ([#d9832fb](https://github.com/pol-sb/ATLAS/commit/d9832fbcb684205c0774a15bb8e75532f9779fc7))

### Refactor

- **core**: implement sanity check for sge computers in toml inputs ([#bfd309e](https://github.com/pol-sb/ATLAS/commit/bfd309e63907ed04c106ffd4a92cddaeb6279605))
- **md**: allow usage of md_apply_temperature_ramp() on benchmark md ([#787ded6](https://github.com/pol-sb/ATLAS/commit/787ded6cdc1fbe1aa42e62aa62278bfaa7a49cc4))

## 0.41.2 (2025-10-09)

### Docs

- **core**: debug documentation.yml ([#bf0a8a5](https://github.com/pol-sb/ATLAS/commit/bf0a8a52807509d673ea621f11251200bfa6eddf))
- **core**: fix 1 missing versions in docs ([#897e3e4](https://github.com/pol-sb/ATLAS/commit/897e3e4c27ebd834df54aa521b9746219e5e44cb))
- **core**: fix redirect in documentation.yml ([#94cdc7b](https://github.com/pol-sb/ATLAS/commit/94cdc7bfbe4d10ccb838172fcf75ba32c10abad0))
- **core**: workflow ([#f4f6a3d](https://github.com/pol-sb/ATLAS/commit/f4f6a3d3d8c76af220d293fee20d9787567040c8))
- **core**: documentation.yml ([#f9bb571](https://github.com/pol-sb/ATLAS/commit/f9bb571a4ef2b4f95070066361c04ac493d997a2))
- **core**: test documentation.yml ([#fd0571f](https://github.com/pol-sb/ATLAS/commit/fd0571f8df6a6c0fe10fd09e3ca1c61aabf9125a))
- **core**: test documentation.yml workflow ([#9723ed5](https://github.com/pol-sb/ATLAS/commit/9723ed5bed56eb031b781a93075102f26f3dd102))
- **core**: test documentation workflow ([#471e070](https://github.com/pol-sb/ATLAS/commit/471e0701038a4def7c03d37521898320cdd88df4))
- **core**: test documentation.yml ([#fae1e59](https://github.com/pol-sb/ATLAS/commit/fae1e59d8c3b930ad437a82210f2b045c3f78a3f))
- **core**: update documentation.yml ([#91bc730](https://github.com/pol-sb/ATLAS/commit/91bc730be802d85aef526147ea05623c93ad0d3c))

### Fix

- **al_loop**: allow al runs to proceed even when no dashboard arg is provided ([#92cdd0a](https://github.com/pol-sb/ATLAS/commit/92cdd0adfff5d4607abb92fb663c286b72d61b9e))
- **al_loop**: change dashboard argument parsing so that al resume runs can run ([#5141670](https://github.com/pol-sb/ATLAS/commit/5141670ff97cfecfec9cd055817df44ec91137bf))

### Misc

- **core**: retry caching in documentation workflow ([#b80c23d](https://github.com/pol-sb/ATLAS/commit/b80c23dc75303b9181e2528f21e5c7c9943a1b9e))
- **core**: test documentation workflow ([#d68f59f](https://github.com/pol-sb/ATLAS/commit/d68f59fad92a62a7bfb3153f2ec36dd02d4d475b))

## 0.41.1 (2025-10-08)

### Fix

- **core**: ntfy.sh was preventing all options not run or resume to work in the atl_active_learning utility ([#ced1b39](https://github.com/pol-sb/ATLAS/commit/ced1b39411613a17f6da8e8262743ae36b638fd5))
- **report**: allow al_loop report to work with latent space exploration in new style AL ([#d49b4e3](https://github.com/pol-sb/ATLAS/commit/d49b4e36aab3b306278d9b01e0da17dddc66f8d4))
- **report**: performance report uses ElapsedRaw instad of CPUTimeRaw to properly compute core*h values ([#5416809](https://github.com/pol-sb/ATLAS/commit/54168097d4964e4cceff7e76a0c496970be80f3d))
- **domain_validity**: update alpha condition to <= 1, fix alpha print in plot ([#c0817ab](https://github.com/pol-sb/ATLAS/commit/c0817ab6e77df1f0d57e9686fd6053af2c4f9782))
- **core**: add optional logger option to custom_print and use it in atl_process_structure.py script ([#7e724a4](https://github.com/pol-sb/ATLAS/commit/7e724a4311bdd68e3a3b5a559e00b18586e482a5))

### Misc

- **core**: update LICENSE and README.md ([#93a0e13](https://github.com/pol-sb/ATLAS/commit/93a0e13a2a1b28a5401184253b13163f08c08366))

## 0.41.0 (2025-10-06)

### Docs

- **schema**: add `frac_points_allowed_out` and descriptions to alpha shape settings in schema ([#41618ff](https://github.com/pol-sb/ATLAS/commit/41618ffd9ee3508d0eddd8c9d1c9f6f5e5500559))
- **schema**: add alpha hull options ([#7d912a2](https://github.com/pol-sb/ATLAS/commit/7d912a2126815b4820e1a783ecb970ce8363ae86))

### Feature

- **domain_validity**: add alpha and area values to alpha hull plot ([#f30e75e](https://github.com/pol-sb/ATLAS/commit/f30e75e56909a092028903a41cbf3bacc5c98648))
- **domain_validity**: add iterative alpha selection during advanced extrapolation check ([#d81436f](https://github.com/pol-sb/ATLAS/commit/d81436fa969be59b12235e0d78e234f8de27eac0))

### Fix

- **al_loop**: check for rmse_e and rmse_f after gathering mace training data to avoid crash ([#20c1b09](https://github.com/pol-sb/ATLAS/commit/20c1b092ea7379a5cd447324805b60af4a471571))
- **report**: load nodes that get passed as np.int instead of orm.Node ([#df3b4d6](https://github.com/pol-sb/ATLAS/commit/df3b4d6c2591d4388f279355b9b405f9e08da9f9))
- **dft**: add blocking schema entry for vasp dft calculator and fix dft log output ([#c966f04](https://github.com/pol-sb/ATLAS/commit/c966f04c5bc1b97aff742619afc72abc89a92a3c))

### Misc

- **core**: update documentation workflow ([#2993e7f](https://github.com/pol-sb/ATLAS/commit/2993e7f1c49afb738bc32bc900405c346f47ed8e))

### Refactor

- **al_loop**: removed unused functions in `simple_active_learning.py` ([#fb51106](https://github.com/pol-sb/ATLAS/commit/fb51106434c68f30017ce36e59f42d2dccda4213))
- **domain_validity**: simplify and remove duplicate functions in `atl_check_descr_combined.py` ([#5d41cfc](https://github.com/pol-sb/ATLAS/commit/5d41cfca4843edb12a103b84fd306693bea4d2d8))

### Style

- **al_loop**: include loop pk in ntfy.sh messages ([#d2feb81](https://github.com/pol-sb/ATLAS/commit/d2feb819f86b683b2f84496566e1b123b0aab57f))
- **al_loop**: separate qr code from logs ([#959cd06](https://github.com/pol-sb/ATLAS/commit/959cd06c0c2bf71424481a754a5f823313e15fc5))

## 0.40.6 (2025-10-02)

### Fix

- **al_loop**: handle missing dashboard arg properly ([#f97860f](https://github.com/pol-sb/ATLAS/commit/f97860f8044deeb05c8a6583dd3ac686f08a2f77))

## 0.40.5 (2025-10-02)

### Fix

- **al_loop**: properly check for args.dashboard in al launch script ([#31b2c4e](https://github.com/pol-sb/ATLAS/commit/31b2c4ed0b8d8f1cbe71efd64dc270f4d722f61a))

## 0.40.4 (2025-10-02)

### Misc

- **core**: fix new documentations.yml github workflow ([#c281dbc](https://github.com/pol-sb/ATLAS/commit/c281dbce7c567d50d5b8c56e913f7361433aad77))

## 0.40.3 (2025-10-02)

### Fix

- **al_loop**: fix crash when active learning is started in resume+submit mode ([#c69410b](https://github.com/pol-sb/ATLAS/commit/c69410b480b5920022c5eddc6e58b15a2a71f741))

## 0.40.2 (2025-10-02)

### Fix

- **al_loop**: add missing debug_mode parameter to resume builder ([#caa14ac](https://github.com/pol-sb/ATLAS/commit/caa14ac41a6801c841d4416d2c15667563b53c86))

### Misc

- **core**: updated git workflow for documentation ([#c19caca](https://github.com/pol-sb/ATLAS/commit/c19caca404715c51fcbdeec81960015b2a4c7ef9))

## 0.40.1 (2025-10-02)

### Docs

- **schema**: separate input from tools ([#caec1a8](https://github.com/pol-sb/ATLAS/commit/caec1a82ebf4970d9ff048d9e7d3fa942bd2e858))
- **schema**: updated schema to include forgotten keys ([#d17d669](https://github.com/pol-sb/ATLAS/commit/d17d6694a7ad064a9b325b51e83a6605eeaabe66))
- **schema**: toml check now detects incorrect keys ([#8c0bbaa](https://github.com/pol-sb/ATLAS/commit/8c0bbaaed6ef3d94d202519b33e8c65e8f92a8b9))

## 0.40.0 (2025-10-01)

### Feature

- **al_loop**: add notification via ntfy.sh to active learning loop ([#417de2e](https://github.com/pol-sb/ATLAS/commit/417de2edef9273cef162e23afc1d7c4ff26d3849))

### Fix

- **al_loop**: fix logging error when submitting calculations to daemon ([#319f3f6](https://github.com/pol-sb/ATLAS/commit/319f3f6d8c96e95b216d7240b5e939be4f0cf726))

## 0.39.1 (2025-09-30)

### Docs

- **schema**: removed obsolete key in schema ([#7dab3cc](https://github.com/pol-sb/ATLAS/commit/7dab3cc8dde626df33d63d017610dcce373590d8))

### Fix

- **report**: improve robustness of report generation when mixing deleted and resumed workchains ([#236fe38](https://github.com/pol-sb/ATLAS/commit/236fe3842b0470f7cec45a50cbe6c64c1cb8c1d2))
- **report**: fixed error capturing in dashboar ([#0885d6e](https://github.com/pol-sb/ATLAS/commit/0885d6edce168f446d7eccfff113f9422bdbe21d))

## 0.39.0 (2025-09-28)

### Docs

- **core**: Improve automatic docs generation and fix wrong info in schema ([#2341927](https://github.com/pol-sb/ATLAS/commit/23419271a158f60ede89cfd128e47494e3cfd29d))

### Feature

- **al_loop**: Decrease scaling of FPS ranking function from O(N^3) to O(N^2) ([#9917c39](https://github.com/pol-sb/ATLAS/commit/9917c39a9f78e23cb2812591342ca395f0b254aa))
- **al_loop**: Add debug mode to alternate between run/submit modes ([#fd6a852](https://github.com/pol-sb/ATLAS/commit/fd6a852f0132dbd43bf58c53f774d12d8b96fb44))

### Fix

- **report**: fixed AL report title ([#7006e8a](https://github.com/pol-sb/ATLAS/commit/7006e8a049d149740c62667e69cafbeace3b70af))
- **domain_validity**: improve performance of autoencoder function by creating tensors from arrays only ([#febe667](https://github.com/pol-sb/ATLAS/commit/febe66774c541e68caed07f6f85a96897f3a6445))
- **init_db**: update diversity metrics ([#b055001](https://github.com/pol-sb/ATLAS/commit/b0550016bc620f6783a882256bebc82c7ffe2717))

## 0.38.0 (2025-09-12)

### Feature

- **domain_validity**: add circles metric and fix vendi score ([#29ee1d3](https://github.com/pol-sb/ATLAS/commit/29ee1d3b3ecefcd0b4cea74e82c499b272f61a80))

### Fix

- **schema**: improve `disagreement_check_type` and `extrapolation_check_type` description ([#2a781ed](https://github.com/pol-sb/ATLAS/commit/2a781ed67a581e16e61bf94d509e11cf7966ce8b))
- **domain_validity**: add n_jobs=-1 to soap descriptor generation ([#bb7599c](https://github.com/pol-sb/ATLAS/commit/bb7599c9efbda96bbd6f10407594e89215928900))
- **al_loop**: changed deprecated usage of node.extras to node.base.extras.all ([#4d1c8d4](https://github.com/pol-sb/ATLAS/commit/4d1c8d468e8f0d2b30f167f174e9eec34dc57542))
- **al_loop**: store working dir in base workchain extras ([#e7be64e](https://github.com/pol-sb/ATLAS/commit/e7be64ec9d2d3e1afd55b348da63e9df701f64cf))

### Style

- **al_loop**: enhance active learning loop reports ([#97755d3](https://github.com/pol-sb/ATLAS/commit/97755d3c7f82867c7405e4245feb0a1de7959086))
- **core**: fix output messages ([#ea761df](https://github.com/pol-sb/ATLAS/commit/ea761df83a57c4f2776272cbfe2f1edd055907e5))

## 0.37.0 (2025-09-10)

### Feature

- **init_db**: first iteration of the metrics library ([#21bc5df](https://github.com/pol-sb/ATLAS/commit/21bc5df8e5c1857d195b6e56e1c3bdd163e09254))

### Fix

- **al_loop**: allow usage of `check_extrapolation_type=none` in active learning ([#5f38975](https://github.com/pol-sb/ATLAS/commit/5f3897581ebebebf7fb02a422d43678aa4d0b324))
- **core**: enable custom_print usage without previous logger enabled ([#96997bb](https://github.com/pol-sb/ATLAS/commit/96997bb9ad38db4546b02cbf54587bcf92fabee4))
- **al_loop**: enable container usage with mace settings.yaml ([#a9c73de](https://github.com/pol-sb/ATLAS/commit/a9c73de2bf4acb188abc300aec0ec0a3d2ffa342))

### Style

- **al_loop**: change run mode to submission, add pretty print ([#51795dc](https://github.com/pol-sb/ATLAS/commit/51795dcc04053b19ca4594b8e716150cf8169a78))

## 0.36.5 (2025-09-09)

### Docs

- **al_loop**: updated inputs ([#2af6bd6](https://github.com/pol-sb/ATLAS/commit/2af6bd6e7106e025ca523ba9ae9bce7516dea721))

### Fix

- **al_loop**: use database path from toml in active learning resume ([#dbfeb49](https://github.com/pol-sb/ATLAS/commit/dbfeb4999d75c69ab408c8f9905261843f184b4b))
- **schema**: update mandatory keys ([#60abf3e](https://github.com/pol-sb/ATLAS/commit/60abf3ee067759db6216312d8993b4d9e5881985))
- **al_loop**: made foundation model optional parameter again ([#3e80ff2](https://github.com/pol-sb/ATLAS/commit/3e80ff29a5f9a2ca8afeb0cb0eeb037def3a3682))

### Misc

- **core**: updated docs template ([#55eb197](https://github.com/pol-sb/ATLAS/commit/55eb197565010bc764659de17487ef0a3257abd9))
- **core**: restore pyproject.toml after testing ([#7d03ec9](https://github.com/pol-sb/ATLAS/commit/7d03ec95d824ee95c6056835427c5f3041f9f763))

## 0.36.4 (2025-09-06)

### Misc

- **core**: update commitizen settings ([#6eec637](https://github.com/pol-sb/ATLAS/commit/6eec6376820bd4861c853bafa7494b1b39678922))

## 0.36.3 (2025-09-06)

### Fix

- **core**: test release script 2 ([#0c1c2b6](https://github.com/pol-sb/ATLAS/commit/0c1c2b633a1e3c78b2669a256134861622c2f6f6))

### Style

- testing new release script ([#755669e](https://github.com/pol-sb/ATLAS/commit/755669ede0ca436c5dc38f87f5a197a0ad0c3881))

## 0.36.2 (2025-09-06)

### Misc

- **core**: update contact info ([#0e977fc](https://github.com/pol-sb/ATLAS/commit/0e977fcb64a498cccc3263b439c537c9b6cb4f17))

## 0.36.1 (2025-09-06)

### Docs

- **schema**: make `dft_method` in active learning mandatory ([#4a1b447](https://github.com/pol-sb/ATLAS/commit/4a1b447045392b47713e714f669128360775522e))

### Fix

- **domain_validity**: pytorch load custom model classes without changing to `weights_only=False` ([#ca05052](https://github.com/pol-sb/ATLAS/commit/ca050523aa6f7cd8e0a64ef56d23d32e86c6a5b4))

### Misc

- **core**: add commitizen templates and missing entries for parsing ([#981c7e2](https://github.com/pol-sb/ATLAS/commit/981c7e2e21f9547727449374b85a20c1462b5f82))

### Style

- **domain_validity**: update concave hull alpha reporting precision ([#d4fa78c](https://github.com/pol-sb/ATLAS/commit/d4fa78c6b950fa46e9462a4c221fb40bbaa32fec))

## 0.36.0 (2025-09-05)

### Feature

- **core**: implement automated documentation generation from schema ([#71f523c](https://github.com/pol-sb/ATLAS/commit/71f523c8a3057e89ec9bb0bd155ef9b4ac0c831c))

### Fix

- **benchmarks**: use global dict to store model data ([#94af2b3](https://github.com/pol-sb/ATLAS/commit/94af2b3936780dbf0d956ce87b22844328fbf1a4))
- **benchmarks**: use global dict to store model name, path, and color to standardize data between plots ([#1efb8ac](https://github.com/pol-sb/ATLAS/commit/1efb8ac6f73d1f0744e14ba08cebdcd2dfe04382))

## 0.35.2 (2025-09-05)

### Fix

- **md**: update schema ([#ab590f6](https://github.com/pol-sb/ATLAS/commit/ab590f67918a38d7fd89be854d82ed2117bac2b4))

## 0.35.1 (2025-09-04)

### Fix

- **init_db**: enable logger and custom prints for toml checking ([#bb64945](https://github.com/pol-sb/ATLAS/commit/bb6494563726f3eb3d88956c5d73449affeaec0d))

## 0.35.0 (2025-09-04)

### Feature

- **core**: Add comprehensive TOML configuration validation system ([#97103bf](https://github.com/pol-sb/ATLAS/commit/97103bf0b6c85d50513f6e909d36e0abcedfd076))
- **core**: implement automated documentation generation from schema ([#e5a4a67](https://github.com/pol-sb/ATLAS/commit/e5a4a67720ca384f26f0dd4d51e8113ab615b5e2))

## 0.34.1 (2025-09-01)

### Fix

- **report**: update log parsing and node gathering during al_loop report generation ([#02d2416](https://github.com/pol-sb/ATLAS/commit/02d2416a4f802ff3c5cd5f23a751922ff7a14b48))
- **report**: readd cueq compatibility for mace ([#ced344d](https://github.com/pol-sb/ATLAS/commit/ced344df2bf055cf1d134e2d5bb9b01e227c1443))
- **report**: improve chart visualization for mlip benchmark ([#23b114a](https://github.com/pol-sb/ATLAS/commit/23b114a53bf4c08b2deee08b747e4e2fc85437a1))

### Misc

- **core**: add new benchmark option for commit s ([#d53e027](https://github.com/pol-sb/ATLAS/commit/d53e027bcb9deb2f70cd3545b5842b79fa18adbd))

## 0.34.0 (2025-08-29)

### Feature

- **al_loop**: first implementation of the data reduction protocol ([#e7f7da6](https://github.com/pol-sb/ATLAS/commit/e7f7da6a91b15790ca848f9f9ce5abd487d453f0))

## 0.33.0 (2025-08-29)

### Feature

- **core**: Add magic cluster benchmark function ([#7ee7fbb](https://github.com/pol-sb/ATLAS/commit/7ee7fbbe19377ffda77965572f98e03364869f6f))

### Fix

- **core**: enable compatibility with new mace finetuning and bump to `mace-torch==0.3.14` ([#5440402](https://github.com/pol-sb/ATLAS/commit/5440402fa2c153d9420b823af15be5091f0df95b))
- **report**: improve chart visualization for mlip benchmark ([#02567f3](https://github.com/pol-sb/ATLAS/commit/02567f35f16773a45a85d56b32c63edb8817d9f1))

## 0.32.0 (2025-08-05)

### Docs

- **core**: Update docs for new `atl_benchmark_mlip` tool ([#cea28d6](https://github.com/pol-sb/ATLAS/commit/cea28d684d22982aff9cd24407b8368284e8cf11))

### Feature

- **al_loop**: Added new benchmark `run_evaluate_database` ([#e74a334](https://github.com/pol-sb/ATLAS/commit/e74a334ec47f5eb434d948f125982ff11b1d24df))
- **al_loop**: added first iteration of benchmark suite for AL-trained MLIP models ([#d99224b](https://github.com/pol-sb/ATLAS/commit/d99224bab8fcd92fe566b991478185fc0aa6ed65))

### Fix

- **core**: Misc fixes and improvements. Added mermaid diagram. ([#25908c3](https://github.com/pol-sb/ATLAS/commit/25908c3de2c0dfd14fcb74bd9b58e151cb204725))
- **report**: display data in dashboard for resumed workchains ([#2cf38d5](https://github.com/pol-sb/ATLAS/commit/2cf38d53d05f8e07a064539b14ca8aeddd79bdcb))
- **dft**: add `ignore_container` option in dft_settings dict ([#db3235a](https://github.com/pol-sb/ATLAS/commit/db3235a7922354b10a4d2b04fa95f6ba6b58f2e2))

### Misc

- **core**: Add entry to `pyproject.toml` for mlip benchmarks ([#0725efc](https://github.com/pol-sb/ATLAS/commit/0725efc6ff8ec776c4e1d53e5b1bcf8d189d26cb))

## 0.31.3 (2025-07-15)

### Fix

- **core**: improve al_loop logging ([#887afdd](https://github.com/pol-sb/ATLAS/commit/887afdd40eab105ff4d000cdb6b3e3a859319417))
- **al_loop**: fix resume mode model loading ([#50276ca](https://github.com/pol-sb/ATLAS/commit/50276ca5f9d44c98c02642694877240bec54ec4c))
- **report**: enable resumed al_loops in dashboard view ([#bf89b40](https://github.com/pol-sb/ATLAS/commit/bf89b4055dd85289b8c6428ace46d33a2a872c2e))

### Misc

- **deps**: bump gunicorn from 22.0.0 to 23.0.0 ([#ed61846](https://github.com/pol-sb/ATLAS/commit/ed61846e17d8d9b87bc108932bc7712d2e3c983a))
- **deps**: bump gunicorn from 22.0.0 to 23.0.0 ([#3ea8534](https://github.com/pol-sb/ATLAS/commit/3ea85347bed0a1ce550afc73504b582b90aa0fff))

## 0.31.2 (2025-07-07)

### Fix

- **core**: update manually version ([#837d2e3](https://github.com/pol-sb/ATLAS/commit/837d2e341e2975a7a3691030cf07381e5639d043))

## 0.31.1 (2025-07-06)

### Fix

- **al_loop**: updated descriptor gathering to use generic `generate_descriptors()` ([#077bd8b](https://github.com/pol-sb/ATLAS/commit/077bd8bdce5e135f14fbd2289cc74351774bf330))

## 0.31.0 (2025-07-06)

### Feature

- **al_loop**: Add ranking algorithms to md seed selection phase ([#4a21367](https://github.com/pol-sb/ATLAS/commit/4a21367b65a071dfb6117b47c6a1232ce2dee5f4))

### Misc

- **core**: update formatting for CITATION.cff in pyproject.toml ([#2beee8a](https://github.com/pol-sb/ATLAS/commit/2beee8af91f84a0882ba4bddb70d591fb69e5a06))

## 0.30.2 (2025-06-20)

### Misc

- **core**: fix broken field in CITATION.cff ([#cef1871](https://github.com/pol-sb/ATLAS/commit/cef18712ba13543310337acc9fa7500850c17f8e))
- **core**: update CITATION.cff ([#19099d5](https://github.com/pol-sb/ATLAS/commit/19099d557a98d984a003fee831a56fbd9fc39792))
- **core**: add first version of `CITATION.cff` ([#a0a8329](https://github.com/pol-sb/ATLAS/commit/a0a8329e7788116bee1c374685a8120dd65a4e9d))

## 0.30.1 (2025-06-06)

NOTE: It seems that documentation commits were not being detected by commitizen changelog up until this point.

- **docs**: several documentation changes.

## 0.30.0 (2025-06-06)

### Feature

- **md**: enable cueq with MD using ASE-MACE ([#0a6c216](https://github.com/pol-sb/ATLAS/commit/0a6c216fbdafad439bec1a4407384f7d85004a6c))

### Fix

- **al_loop**: improve robustness of `TrainMACEModelCalculationParser` ([#2336d8c](https://github.com/pol-sb/ATLAS/commit/2336d8c38b9968471729710d2d62935649726a94))
- **al_loop**: fix `mace-training-parser` as the parser for mace training calculations ([#f1d49a3](https://github.com/pol-sb/ATLAS/commit/f1d49a34526f7269de851bf86b6a941001d02f35))

## 0.29.4 (2025-06-05)

### Misc

- **core**: bumped up mace-torch version from 0.3.7 to 0.3.12 ([#e2b653f](https://github.com/pol-sb/ATLAS/commit/e2b653fb8da4cb3cce1afcca5d5cc37e7e4aead9))

## 0.29.3 (2025-06-02)

### Fix

- **al_loop**: improve argument parsing for model training ([#a22ead7](https://github.com/pol-sb/ATLAS/commit/a22ead7c55b4678189fe1c463884d10e35a19c36))
- **report**: enable loading autoencoder for al_loop report ([#13baca3](https://github.com/pol-sb/ATLAS/commit/13baca3c60657401033293cce29aebf7b0255cc9))

## 0.29.2 (2025-05-27)

### Fix

- **al_loop**: allow gathering of num_threads during active learning md setup ([#06cba1d](https://github.com/pol-sb/ATLAS/commit/06cba1d7c913e45beea3d2160ffc84b70658a450))

## 0.29.1 (2025-05-27)

### Fix

- **md**: read num_cpus setting properly from MD input file ([#13a8568](https://github.com/pol-sb/ATLAS/commit/13a8568a33edce8148c244ccaac326ce2170dfb7))
- **dft**: add correct default values for CalcType and DFT calc builder ([#d7fbc15](https://github.com/pol-sb/ATLAS/commit/d7fbc15d4c308f0b89ff66142b33bb4249125078))

### Misc

- **core**: clean pyproject and add pip dependency ([#c94a58e](https://github.com/pol-sb/ATLAS/commit/c94a58ebfceefdb8d907fd4630d560c470f89167))

## 0.29.0 (2025-05-26)

### Feature

- **domain_validity**: Updated concave hull algorithm with python implementation and automatic alpha selection ([#e2f904b](https://github.com/pol-sb/ATLAS/commit/e2f904b0c95ffe34a53a3a653e481f9d5328d9d3))

### Fix

- **al_loop**: better tracking of `atl_al_step` in database during active learning runs ([#7858443](https://github.com/pol-sb/ATLAS/commit/78584438f630e59d003f10196d977c5a4ad8e5e6))
- **core**: allow loglevel below 20 to show in stdout ([#c337e1a](https://github.com/pol-sb/ATLAS/commit/c337e1a0691a57529d9e133b0d4bb0a606c80b34))
- **report**: enable latent space exploration plot for runs without latent space in workchain ([#c9267d4](https://github.com/pol-sb/ATLAS/commit/c9267d484157aec706f3ba2f9c030ab8b0f53b97))
- **al_loop**: modify tracking of current step in initial database ([#d01df1d](https://github.com/pol-sb/ATLAS/commit/d01df1d2c39a056f0ca0566bea53e94ae03162e6))
- **md**: improve extrapolation robustness and compatibility in `atl_process_structure.py` script ([#866d165](https://github.com/pol-sb/ATLAS/commit/866d16584f94f0483a8590daf2e64f1f539db814))

## 0.28.3 (2025-05-23)

### Fix

- **report**: update autoencoder report generation ([#8e7cb64](https://github.com/pol-sb/ATLAS/commit/8e7cb649abfa4c48b5fab44696525bcc0f172117))
- **al_loop**: ensure compatibility with aiida-vasp v2 workflows and enable partial relaxation ([#3249a3a](https://github.com/pol-sb/ATLAS/commit/3249a3a603ff452dc6edebb785f71dc515d38314))

### Misc

- **core**: add pdb dependency in dev install optional ([#daca7a2](https://github.com/pol-sb/ATLAS/commit/daca7a2b0e31150a494b22171af2b8a13c64547e))

## 0.28.2 (2025-05-19)

### Fix

- **report**: removed quit() statement blocking al_loop report from saving images ([#041cd27](https://github.com/pol-sb/ATLAS/commit/041cd279972f2b1b4d24bf9e1e67db9321e5be75))

## 0.28.1 (2025-05-18)

### Fix

- **al_loop**: Add check for inf or nan values in `initial_magmoms` ([#f098089](https://github.com/pol-sb/ATLAS/commit/f0980897171f7e3bcaaf4df9f1aca759d14abcb6))
- **report**: add missing `NonExistent` import ([#bc25cd2](https://github.com/pol-sb/ATLAS/commit/bc25cd25679ccb1361f97d060ec9f87883815062))

## 0.28.0 (2025-05-17)

### Feature

- **md**: Added `ATLSafeCalculatorWrapper` to wrap ASE MD calculations and detect unphysical energies or forces. ([#774bf52](https://github.com/pol-sb/ATLAS/commit/774bf52709cbe3ba4f25fc893658b893fa320cef))

## 0.27.6 (2025-05-16)

### Fix

- **al_loop**: modified high T check in `apply_filter_exploding_structures()` for T_list usage ([#f9842ce](https://github.com/pol-sb/ATLAS/commit/f9842ce7af0c1b2c0d7d5cd9ebe1a9b0e1baad2a))
- **al_loop**: renamed incorrect `log_interval` to `loginterval` for ase calculator ([#d3a47e7](https://github.com/pol-sb/ATLAS/commit/d3a47e7776e621064aca149a517a1bd3422e4be5))
- **al_loop**: write descriptors for every structure in `descriptor_dict`. ([#a85053d](https://github.com/pol-sb/ATLAS/commit/a85053ded59db7c0d5e636fddf99a87bc40983e4))

### Misc

- **core**: added 'md' scope to commitizen template ([#a559b72](https://github.com/pol-sb/ATLAS/commit/a559b7221c53c44e0b8e5404817e99f0873e8fef))
- **core**: updated authors ([#2b4a5a2](https://github.com/pol-sb/ATLAS/commit/2b4a5a263ba9b4698c800622ab97ae5f59cd118c))

## 0.27.5 (2025-05-15)

### Fix

- **al_loop**: enable usage of `log_save_interval` in md code ([#c365f65](https://github.com/pol-sb/ATLAS/commit/c365f65a53bbc8a1468cab60bcd510d3393e69f2))
- **al_loop**: updated dft calculator settings in al_loop for aiida-vasp==4.1.0 ([#dfa11d3](https://github.com/pol-sb/ATLAS/commit/dfa11d3cec55a9eef10811b0b030d8bc2ab6fc5e))

### Misc

- **core**: add emoji to commitizen scope descriptions ([#14b8611](https://github.com/pol-sb/ATLAS/commit/14b8611587eb94b7176614ffae4914ade28d120f))

## 0.27.4 (2025-05-15)

### Misc

- **core**: clean up pyproject.toml ([#75f12fa](https://github.com/pol-sb/ATLAS/commit/75f12fab473675a675631309478e6b92928cd1fb))
- **core**: update commitizen schema ([#d7b8c24](https://github.com/pol-sb/ATLAS/commit/d7b8c2430c9af2838eb64e99d9148f8b740e509c))

## 0.27.3 (2025-05-15)

### Fix

- **report**: add table view for al_loop and performance reports ([#bed2aa5](https://github.com/pol-sb/ATLAS/commit/bed2aa597778acd590e565b3e8af2e26733c67ee))
- **al_loop**: fixed report typo ([#817486a](https://github.com/pol-sb/ATLAS/commit/817486ae4da550da5755b18641ea03bb2f13676f))
- **al_loop**: commented unused version of exploding structure filters. development will be moved to a branch ([#c9372ed](https://github.com/pol-sb/ATLAS/commit/c9372edfa7ce56be86b1f67fae9cfdb2ab4610e7))
- **report**: remove debug breakpoints ([#3318f95](https://github.com/pol-sb/ATLAS/commit/3318f953ba4be1e27432b7b75455353dc0730aa8))

## 0.27.2 (2025-05-14)

### Fix

- **al_loop**: change `cov_rad_multiplier_min` default to `0.8` and update docs ([#6d4a1c9](https://github.com/pol-sb/ATLAS/commit/6d4a1c9a788c39706673e0a3364c294d9d03575c))

## 0.27.1 (2025-05-14)

### Fix

- **report**: move iteration limit function inside plot function ([#b3f448c](https://github.com/pol-sb/ATLAS/commit/b3f448c88f5616050f33db4ec433ae616557cbdc))
- **al_loop**: allow correct builder code gathering for aiida-vasp cal... ([#e21ad69](https://github.com/pol-sb/ATLAS/commit/e21ad69e57b2a3341c7da4196afe2a4862eb62a2))
- **al_loop**: allow correct builder code gathering for aiida-vasp calculations ([#6f1a863](https://github.com/pol-sb/ATLAS/commit/6f1a8635f6ae8ee01e3779f0d1ab5bf3234feeeb))
- **report**: actually use length of xaxis to determine barchart width ([#44893e2](https://github.com/pol-sb/ATLAS/commit/44893e2958c9116ccc0d813914b68f78da3ac696))
- **report**: improve auto assignation of barchart width ([#e9d5b18](https://github.com/pol-sb/ATLAS/commit/e9d5b1849bb3a98260646a1426ee804ec14c8154))
- **report**: fix usage on instance of new `get_loop_report` ([#b8b6f61](https://github.com/pol-sb/ATLAS/commit/b8b6f61fc16f302b727d0b28a898c7423b782f9d))

## 0.27.0 (2025-05-14)

### Feature

- **report**: new version of `get_loop_report` and allow limiting number of steps shown in plot by `limit_num_steps` ([#f6b1820](https://github.com/pol-sb/ATLAS/commit/f6b1820caa6fbb2048bbaa7f9ab25c0b0cc58c69))
- **al_loop**: add `sample_frames_during_md` to allow to sample md on the fly ([#ba8a393](https://github.com/pol-sb/ATLAS/commit/ba8a39342926e3769ebc79d16fac3bcffdb4236f))

### Fix

- **al_loop**: improve robustness of uuid labelling for new structures ([#21496c4](https://github.com/pol-sb/ATLAS/commit/21496c40e69f6719dea29626cf963ddfe1d34de7))
- **al_loop**: update builder code check to ensure compatibility with `aiida-vasp==4.1.0` ([#109768d](https://github.com/pol-sb/ATLAS/commit/109768d4235245449e47e9dc3a1931020c846adb))
- **al_loop**: allow to change `md_stop_explode_filter` interval via new `explode_check_interval_perc` ([#67d83b4](https://github.com/pol-sb/ATLAS/commit/67d83b44150a6b9d165536866a29a4ebfb1d2e6c))

## 0.26.2 (2025-05-13)

### Fix

- **core**: update aiida-vasp dependency ([#195a283](https://github.com/pol-sb/ATLAS/commit/195a2835d600589e73df463c0c8cb71f93e2aff6))
- **dft**: remove debug print ([#f01a40e](https://github.com/pol-sb/ATLAS/commit/f01a40eb9f3da490cc282b4727ec271a1e3696b0))

## 0.26.1 (2025-05-12)

### Fix

- **al_loop**: allow steps with no DFT calculations to continue without throwing exception ([#23d84fd](https://github.com/pol-sb/ATLAS/commit/23d84fdad9f4b10c29fb4927e27e205960aa9a5f))

## 0.26.0 (2025-05-07)

### Feature

- **al_loop**: add optional threshold filter for new NN-DFT results - Implemented `filter_dft_calcs_threshold` function to compare new DFT   results against prior NN predictions. - Modified `get_dft_calc_builder_mace_list` to ensure prior NN predictions   are stored as `curr_model_energy`/`curr_model_forces` on structures   before they are submitted for DFT. - Integrated this filtering step into the `SimpleActiveLearningWorkChain`   after DFT calculations are gathered and before they are added to the   training database. - Added new configuration options (`filter_dft_calcs`, `threshold_E_meV`,   `threshold_F_meV`) under the `dft.mace.filters` section in   `active_learning_settings.toml` and updated documentation in `input.md`. - Refactored `gather_dft_calcs_mace` to return a list of serialized ASE   Atoms objects instead of a file path, facilitating in-memory processing   such as the new filtering step. - Minor formatting adjustments in documentation. ([#88cfe57](https://github.com/pol-sb/ATLAS/commit/88cfe57612dd54dbbd34fdc42dea300de07ef3fd))
- **report**: Add AL performance reports and revamp docs ([#b0f0049](https://github.com/pol-sb/ATLAS/commit/b0f0049c685e6cd0f262de120535c0ebeef05f3a))

### Fix

- **core**: updated aiida utils to enable compatibility with VASP DFT structure relaxation ([#2ea059e](https://github.com/pol-sb/ATLAS/commit/2ea059e06f8059cf9a31def2680a1381f940cf1b))
- **init_db**: replaced old `lattice_frac_displ_` keys with `lattice_frac_deform_` key usage ([#0df0563](https://github.com/pol-sb/ATLAS/commit/0df0563a53b650ea0ce435f547864cd948e372c8))

## 0.25.3 (2025-04-22)

### Fix

- **init_db**: change series row indexing to use .iloc to ensure compatibility with newer Pandas versions ([#b731c7a](https://github.com/pol-sb/ATLAS/commit/b731c7a079044d7ec38dec7cd05b26fa38f2f2e1))
- **core**: update model dtype before loading when using MACECalculator ([#ac1dc4a](https://github.com/pol-sb/ATLAS/commit/ac1dc4a170b8e2d9879167ef8f38c280079cb0ff))
- **core**: ensure assignation of `atl_id` key for unique id during al_loop, md and database generation ([#3e494ce](https://github.com/pol-sb/ATLAS/commit/3e494ceb6d33836fbad09835f12990e9898f4898))

## 0.25.2 (2025-04-17)

### Fix

- **dft**: allow relaxation on mdb vasp dft ([#438eee0](https://github.com/pol-sb/ATLAS/commit/438eee097eec74b21b7cefb3d426cf879384187e))
- **init_db**: improve initial database report generation ([#4525418](https://github.com/pol-sb/ATLAS/commit/4525418745c2d02b7401b42791553d0811c9737f))
- **al_loop**: save md uuids as string to enable ase saving ([#c53f801](https://github.com/pol-sb/ATLAS/commit/c53f801b19e4b3bd7272d53a631a6728d65e30f4))
- **db_gen**: improve argparse speed by moving imports ([#f0689d6](https://github.com/pol-sb/ATLAS/commit/f0689d6e0fee1408a70b0f8218b4a14f0ce86435))
- **al_loop**: fix al_loop report gathering for old loops ([#495db6b](https://github.com/pol-sb/ATLAS/commit/495db6b22d8e7701ceec1805c02e367744c36c91))
- **al_loop**: fix compressed file extension ([#632c5b4](https://github.com/pol-sb/ATLAS/commit/632c5b477b2ad360c447ae42adeee996d15dd148))
- **al_loop**: fix compressed file extension ([#77f635e](https://github.com/pol-sb/ATLAS/commit/77f635e2bd39f96ff59b265aedc020ef54fbbbf0))
- **al_loop**: remove IsolatedAtom from the seed databases ([#39d7f06](https://github.com/pol-sb/ATLAS/commit/39d7f067cd8e9ccb46e344988e0ab88c30c629cc))
- **al_loop**: check if resume dir is a directory ([#19a3ad1](https://github.com/pol-sb/ATLAS/commit/19a3ad1aa2b4384378e4888959d9dfa8396d5095))

## 0.25.1 (2025-04-04)

### Fix

- **al_loop**: change value for default parameters on md explosion filters ([#6cb6ca1](https://github.com/pol-sb/ATLAS/commit/6cb6ca1889e8982fd89e527e882986e8abfc7965))

## 0.25.0 (2025-04-03)

### Feature

- **al_loop**: add high T and positive E check to explode MD filter. ([#d5f3e49](https://github.com/pol-sb/ATLAS/commit/d5f3e49b335ff7cc8566f35ee0d5b25d048261aa))

## 0.24.2 (2025-04-03)

### Fix

- **al_loop**: use atl_id key instead of aiida_uuid for uuid ([#facd888](https://github.com/pol-sb/ATLAS/commit/facd888d567bdea955c7b0e2143e653952187844))
- **al_loop**: fix repetated uuids during db generation and active learning loop ([#9a5e612](https://github.com/pol-sb/ATLAS/commit/9a5e61220e16b802610022a818e52bd042abd549))

## 0.24.1 (2025-04-02)

### Fix

- **al_loop**: adresses #99. remove unused import ([#6203e43](https://github.com/pol-sb/ATLAS/commit/6203e4355001d09ea081edf8041328d96b202a3d))
- **al_loop**: adresses #99. Raise error when `arr_0` not found. ([#9334eaa](https://github.com/pol-sb/ATLAS/commit/9334eaadf94da53950d8bfd4285bc3b9931f2ad2))
- **al_loop**: correctly load npz files for descriptors ([#797f31c](https://github.com/pol-sb/ATLAS/commit/797f31cceb9618848fe1d850d9b897f4bc36da91))
- **al_loop**: fix descriptor file name assignation ([#3d48ffd](https://github.com/pol-sb/ATLAS/commit/3d48ffdc136277686d21eb7412de96e79a7a4df8))

## 0.24.0 (2025-04-01)

### Feature

- **gen_db**: add report option to `atl_gen_init_db` script ([#1bb63a4](https://github.com/pol-sb/ATLAS/commit/1bb63a4585b6855ed3d73f02ffa958ec9b00728c))
- **gen_db**: add user-facing `load_database` function to allow loading generated databases ([#271d2a1](https://github.com/pol-sb/ATLAS/commit/271d2a1f341423561feb0e2b691dc78a876dc92e))

### Fix

- **al_loop**: fix model conversion before descriptor generation ([#b21f773](https://github.com/pol-sb/ATLAS/commit/b21f7735d4d33035f2e4b801e7390b52bc7bd493))

## 0.23.1 (2025-03-30)

### Fix

- **al_loop**: fix container usage in simple active learning loop ([#674ff77](https://github.com/pol-sb/ATLAS/commit/674ff77556c461e1bc83f648840c87ab703174c1))
- **al_loop**: include `--save_cpu` option to allow exchange between gpu/cpu trained MACE models ([#dbaa0ef](https://github.com/pol-sb/ATLAS/commit/dbaa0ef15899f01e67e7fa53f97d2d9d3dc444ec))
- **al_loop**: correctly evaluate existence of traj_obj so `md_write_frame_traj` function can be attached to md ([#64e9c57](https://github.com/pol-sb/ATLAS/commit/64e9c57fe31ad2e51fb4e27177bc52d5cd675216))

## 0.23.0 (2025-03-29)

### Feature

- **al_loop**: use compressed numpy arrays for descriptor storage ([#1354ccb](https://github.com/pol-sb/ATLAS/commit/1354ccb25d37949fb10874eedfcc8d6997c6e314))
- **al_loop**: enable optional container usage for every step in the AL Loop ([#9cef32d](https://github.com/pol-sb/ATLAS/commit/9cef32dc63f2310d4f6efe5ba78a1c648b530a4f))
- **db_gen**: add MD frame generation for initial database creation ([#188245c](https://github.com/pol-sb/ATLAS/commit/188245c0885d9ef95ad1ea277a1d46439cbd6fd5))
- **db_gen**: add `to_ase_atoms` function and `init_md` identifier ([#94f92c8](https://github.com/pol-sb/ATLAS/commit/94f92c829d48950e655d4c7d4317d0172c38ae5c))
- **al_loop**: add exploding structures filter ([#4d044bb](https://github.com/pol-sb/ATLAS/commit/4d044bbcee14f6877eac2a0d0e8f591b245124e1))

### Fix

- **init_db**: add `init_md=False` to new IsolatedAtom configurations ([#310cf43](https://github.com/pol-sb/ATLAS/commit/310cf438d16a765a3f1140f5de4d6c5a4fc230ad))
- **al_loop**: update thresholds for `atl_process_structure.py` and fix wrong keys ([#323492e](https://github.com/pol-sb/ATLAS/commit/323492ec171a331b9c7fa85594cb8e3fdb06c229))
- **al_loop**: improve database filtering in `apply_filters_db` ([#58999c8](https://github.com/pol-sb/ATLAS/commit/58999c863236a5a7f96f7de4a43ebe8d0e43d409))
- **al_loop**: enable container usage for mace tools ([#13a7d97](https://github.com/pol-sb/ATLAS/commit/13a7d97c25ae0e3985ec88f343644270141b5772))
- **al_loop**: properly use enable_cueq flag in mace training ([#6248a94](https://github.com/pol-sb/ATLAS/commit/6248a94f0c3ecd184385d313e5e2084f7b812aac))

## 0.22.2 (2025-03-20)

### Fix

- **al_loop**: improve legend in al_loop report ([#0692270](https://github.com/pol-sb/ATLAS/commit/06922708af0ca97f03cf6f5903ebc6034fc6aa9a))
- **core**: add num_jobs check to can_submit_calculation ([#15b0da3](https://github.com/pol-sb/ATLAS/commit/15b0da30d3ae835ac882a35b106e748b897ec4fd))
- **al_loop**: make MD calculator units consistent. Zero linear and angular momentum ([#78c455e](https://github.com/pol-sb/ATLAS/commit/78c455ed290af13fcee5789b253f72a4116f18fc))
- **core**: pin mp_api requirement to at least v0.45.3 ([#45a96c0](https://github.com/pol-sb/ATLAS/commit/45a96c0b22051ef490108b9f5cafbbe393f4a90a))

## 0.22.1 (2025-03-18)

### Fix

- **al_loop**: update md temperature during MD run ([#2f29bb9](https://github.com/pol-sb/ATLAS/commit/2f29bb96e3934fded4639ffd9eb83c7ef79597ac))

## 0.22.0 (2025-03-18)

### Feature

- **core**: add unified structure filtering in core.filtering ([#5c29fac](https://github.com/pol-sb/ATLAS/commit/5c29facb9edbf08c3e0df92d2cd73b86c7c441f3))

## 0.21.1 (2025-03-17)

## 0.21.0 (2025-03-17)

### Feature

- **gen_db**: add structure filtering during database generation ([#4bf77a5](https://github.com/pol-sb/ATLAS/commit/4bf77a59b6b6503f8969da1fffe8df087c8aab5e))

### Fix

- **al_loop**: restructure md inputs and add covalent multiplier input parsing ([#6c6f0e8](https://github.com/pol-sb/ATLAS/commit/6c6f0e85a597b1253963b0db8af50e223410ae35))
- **al_loop**: add wrapping + z axis offset to MD filters. Include covalent multiplier ([#35d824c](https://github.com/pol-sb/ATLAS/commit/35d824cc644346202649b65530a3384c3fb271d0))
- **al_loop**: use sorted file data to resume calculations ([#ef8d8d0](https://github.com/pol-sb/ATLAS/commit/ef8d8d08dacd934a3573028cd4306f498e283838))
- **al_loop**: use sorted file data to resume calculations ([#d77fd76](https://github.com/pol-sb/ATLAS/commit/d77fd76a513ca86645af7d7896d8c2ae707a291a))
- **al_loop**: add `aiida_wait_submit` function and fix submission queue ([#f99ee24](https://github.com/pol-sb/ATLAS/commit/f99ee24167e15d25d0baee848f426c23ec5bfdd5))
- **al_loop**: decrease `maximum_value_f` multiplier to 500 ([#253b5c6](https://github.com/pol-sb/ATLAS/commit/253b5c6b5af2c836255fa57b9468750f02e93310))
- **gen_db**: actually assign bulk INCAR for default cases ([#d5f9abc](https://github.com/pol-sb/ATLAS/commit/d5f9abcebc7423493d94a07613622e0859131b5e))
- **al_loop**: ensure that models are loaded only for the first resumed step ([#7b7f8f2](https://github.com/pol-sb/ATLAS/commit/7b7f8f2f08d43766d935bf59c99725ec7e1d12af))

## 0.20.21 (2025-03-05)

### Fix

- **al_loop**: fix `energies_stat` being used for `f_mean_error` and `f_std_error`. ([#3dcd2c5](https://github.com/pol-sb/ATLAS/commit/3dcd2c5aaf6c8ca8888641e713d69d376d98c080))
- **gen_db**: improve database generation output ([#c8afc66](https://github.com/pol-sb/ATLAS/commit/c8afc6662eff4e63bf52a7694829995f46927715))
- **al_loop**: gather resume iteration number properly + make mace_potential_path only mandatory for dft mace ([#84a362a](https://github.com/pol-sb/ATLAS/commit/84a362a0ca9ffa20f11cb1e00f41caef98cbc640))
- **al_loop**: add `al_start_mode` input to simple active learning to allow loading already trained MLIP models ([#0959c7d](https://github.com/pol-sb/ATLAS/commit/0959c7d166edffaa4f0ce4a84965c51b0bd5c984))
- **core**: updated database generation template ([#2e1358e](https://github.com/pol-sb/ATLAS/commit/2e1358e2a63e1257e76de3df9243bab5a24776d3))
- **core**: fix log generation ([#9a019a8](https://github.com/pol-sb/ATLAS/commit/9a019a8e84a9ba466b5a019878dac237a577eafb))

## 0.20.20 (2025-02-24)

### Fix

- **al_loop**: update status gui to be compatible with simple active learning ([#b2a64e4](https://github.com/pol-sb/ATLAS/commit/b2a64e4bdfb0dc4a825a8b2985331a326e28a09a))
- **al_loop**: fixed al_loop report and added svg saving ([#d210525](https://github.com/pol-sb/ATLAS/commit/d210525ff153c55146e31106aa763eaccaa9a841))
- **al_loop**: fixed extrapolation_plot placeholder ([#c4724ec](https://github.com/pol-sb/ATLAS/commit/c4724ec1a637226bcd152a06c4bb0c906635c696))

## 0.20.19 (2025-02-23)

### Fix

- **al_loop**: create empty temporary file for missing `extrapolation_plot` file ([#c92f5b8](https://github.com/pol-sb/ATLAS/commit/c92f5b8d5c9145871581a3e7f3346a1f69e33202))
- **al_loop**: make al_loop report compatible with new F_max/F_avg and fix labels ([#fe0c7d8](https://github.com/pol-sb/ATLAS/commit/fe0c7d89bcc6627933362a50ec77942e6feb9638))
- **al_loop**: improved logged information ([#e4ce21c](https://github.com/pol-sb/ATLAS/commit/e4ce21cacd6aa3f0e898a94064174593ad32bb82))
- **al_loop**: remove from seed_db processing calculations that finished with errors ([#8101e5a](https://github.com/pol-sb/ATLAS/commit/8101e5a6f62e9c60900580203c1806df069426a2))
- **al_loop**: fix outlier masking ([#658591b](https://github.com/pol-sb/ATLAS/commit/658591b6d052966224b261ab114eb974587b1201))
- **al_loop**: fix args for per_atom option in init_db plot ([#9b446bf](https://github.com/pol-sb/ATLAS/commit/9b446bf4465fc90c37aaadc573ed94d4f6cc19de))
- **al_loop**: add per_atom option to init_db plot ([#621ba68](https://github.com/pol-sb/ATLAS/commit/621ba68467034b3bd0d366379ed8034e60e06cf4))
- **al_loop**: energy filtering uses values directly, to filter high energies instead of low ones ([#a8e0d03](https://github.com/pol-sb/ATLAS/commit/a8e0d03990cd3f23f9dca9c8d434dfd9b899d445))
- **al_loop**: change std. dev. to force max in init_db report ([#bc498ca](https://github.com/pol-sb/ATLAS/commit/bc498caff63ae92df010bd7f299df6638f22bc25))

## 0.20.18 (2025-02-19)

### Fix

- **init_db**: relabel custom_format function ([#6524f99](https://github.com/pol-sb/ATLAS/commit/6524f99a0dfa5acfc3a34061933fc4427b883cb2))
- **al_loop**: add `show_log_path` option to `init_logger()` ([#820b248](https://github.com/pol-sb/ATLAS/commit/820b248c8f9e62313e0de4406688e14fa3ff5b92))
- **al_loop**: make init_db report interactive ([#02f2c24](https://github.com/pol-sb/ATLAS/commit/02f2c244ef26bffba2cc08b5cffa1dc6b7ffd3a4))

## 0.20.17 (2025-02-18)

### Fix

- **al_loop**: add forces to report ([#490143c](https://github.com/pol-sb/ATLAS/commit/490143c3c121a66fef2c988e0e7a9934ac5430ec))

### Misc

- **docs**: fixed typo on `input.md` ([#032b346](https://github.com/pol-sb/ATLAS/commit/032b34662652235f15bbc0ab42b997686bac5e85))

## 0.20.16 (2025-02-17)

### Fix

- **al_loop**: cleanup ([#a2afb30](https://github.com/pol-sb/ATLAS/commit/a2afb3060953202b02fcc56b45e45b5f8bad4a13))
- **al_loop**: fixed units and statistics for `training` mode in `atl_process_structure.py` ([#f8e28b6](https://github.com/pol-sb/ATLAS/commit/f8e28b6d2cb84112f44d131d9ce6de96eb705e32))
- **al_loop**: updated threshold multiplier for forces for `md_threshold` mode in `atl_process_structure.py` ([#c31547b](https://github.com/pol-sb/ATLAS/commit/c31547b171a870ae12f68fcb761f5aaa0bada107))
- **al_loop**: added empty dft calcs so iterations that remove all dft structuer dont crash ([#39a940c](https://github.com/pol-sb/ATLAS/commit/39a940cd3923c8c9934a1a42ccfa50f28002a96b))

### Misc

- **al_loop**: added additional log messages to `atl_process_structure.py` ([#664f4d5](https://github.com/pol-sb/ATLAS/commit/664f4d55941a6dc10c4752ceb07c625dc28bfce8))

## 0.20.15 (2025-02-15)

### Fix

- **al_loop**: re-added training-based interpolation strategy ([#71513de](https://github.com/pol-sb/ATLAS/commit/71513decaad40a71046f32f5bdff9a5edd31633f))
- **al_loop**: add extrapolation detailed logs to `atl_process_structure` ([#5d3dfe1](https://github.com/pol-sb/ATLAS/commit/5d3dfe1bfa732474b49e70357de5671d0e4780b9))
- **al_loop**: add concave hull to report ([#849dbd0](https://github.com/pol-sb/ATLAS/commit/849dbd03109ca3cffd4a7746ca94405db1c8c2b3))
- **al_loop**: re add `none` extrapolation type, so only E and F are checked with the committee ([#be7b6d5](https://github.com/pol-sb/ATLAS/commit/be7b6d5c97be1be14b582e061e1f9089f1d102f1))

### Misc

- **al_loop**: function cleanup ([#64936e2](https://github.com/pol-sb/ATLAS/commit/64936e2c697416111312265499142f0f66bd6f66))

## 0.20.14 (2025-02-12)

### Fix

- **al_loop**: update `al_loop` to `atl_al_loop` in complex al loop version ([#98443c6](https://github.com/pol-sb/ATLAS/commit/98443c6bd74cfddb6da259133cf45da67263a0b2))
- **al_loop**: improve representation of database evolution through latent space ([#f1dbda2](https://github.com/pol-sb/ATLAS/commit/f1dbda2df8f7f3ab944fb4b9b9d425a987aadbb0))
- **al_loop**: add active learning loop step index to new structures in database ([#c4e9d74](https://github.com/pol-sb/ATLAS/commit/c4e9d74053c16f7d82977afa7c6f41341618764f))
- **al_loop**: correctly gather energy and forces from vasp calculations ([#9a2f5ed](https://github.com/pol-sb/ATLAS/commit/9a2f5edb3516ad0674a66f391f24de3b53a7a478))
- **al_loop**: update keys to REF_ prefix ([#e7a4a9d](https://github.com/pol-sb/ATLAS/commit/e7a4a9df39cfb82fc72c377d20abc436c88320ef))
- **al_loop**: allow using a single autoencoder for the entire loop ([#d94ae84](https://github.com/pol-sb/ATLAS/commit/d94ae84ec47c2b71ddcbd9b6c55b07ed99897e16))
- **al_loop**: simplified MACE evaluation with report ([#da7117d](https://github.com/pol-sb/ATLAS/commit/da7117dfff8cbe9d82d1c6aafaa2fbb5d9bb7470))
- **al_loop**: replace atl_process_structure extrapolating frames keys with REF_xxx keys usable with MACE ([#b857216](https://github.com/pol-sb/ATLAS/commit/b85721671ef07983d0ab3a9db139b407066fd491))

### Misc

- **al_loop**: improved output comments for MACE evaluation during active learning loop ([#5ca16de](https://github.com/pol-sb/ATLAS/commit/5ca16de67f04dbf7ed11d27879d87448f0924362))
- **al_loop**: improved output comments ([#b3801e9](https://github.com/pol-sb/ATLAS/commit/b3801e95b0d15e9e0f80282c495eb57ffa5584bb))
- **core**: remove unused comments ([#6014efa](https://github.com/pol-sb/ATLAS/commit/6014efa36a893803be066b57780b5508ef23b94f))
- **core**: add `none` print type to `custrom_print` function ([#26eebc3](https://github.com/pol-sb/ATLAS/commit/26eebc34ec9e319e7364a8a1818e4b4b859d00b7))

## 0.20.13 (2025-02-06)

### Fix

- **al_loop**: replace default calculator keys with REF_xxx keys usable with MACE ([#eee81ee](https://github.com/pol-sb/ATLAS/commit/eee81eef03813411a6cde0d5e691a5913f5e01e5))
- **al_loop**: replace default calculator keys with REF_xxx keys usable with MACE ([#b46542f](https://github.com/pol-sb/ATLAS/commit/b46542ffec9dca86085e16f4cd4889fd16dd80c5))
- **al_loop**: added missing_ok to unlink call, to avoid error on tmp file removal ([#d9d44e7](https://github.com/pol-sb/ATLAS/commit/d9d44e7a0741d6e8309acf8919ee7df8a33d57ad))
- **al_loop**: use file path for tmp file unlink instead of wrong `_TemporaryFileWrapper` object ([#4bb20ac](https://github.com/pol-sb/ATLAS/commit/4bb20ac055ad192cc204d971a6f5a2725f13fe1a))

### Misc

- **al_loop**: removed unused comments ([#aa90b81](https://github.com/pol-sb/ATLAS/commit/aa90b818ee7220913cfc4d699b48803c5874d673))
- **al_loop**: added missing type in docstring ([#59c491b](https://github.com/pol-sb/ATLAS/commit/59c491ba32d6c0c8d7fe3e29644e6234aa0ecdcd))
- **al_loop**: added toggle for multihead finetuning ([#25c03a6](https://github.com/pol-sb/ATLAS/commit/25c03a63d0a2e7013359251e2fd7c5f3af8eb4f8))

## 0.20.12 (2025-02-06)

### Misc

- **al_loop**: remove isolatedatoms configs from error plot ([#93c55a5](https://github.com/pol-sb/ATLAS/commit/93c55a5c8f064df2800254d6210c57224c74a74d))
- **al_loop**: fixed typo in comments ([#4edcfb0](https://github.com/pol-sb/ATLAS/commit/4edcfb03671a40f0608b4a16d992e2bd98c069a2))
- **al_loop**: added toggle for multihead finetuning ([#7d71d03](https://github.com/pol-sb/ATLAS/commit/7d71d035180b721c3f0b55a93229ec49c1ad49a2))
- **al_loop**: added file cleanup ([#01447bb](https://github.com/pol-sb/ATLAS/commit/01447bbfef3d82bc18cc8b06aefcb8a990327bcb))

## 0.20.11 (2025-02-03)

### Fix

- **dft**: add function to check direction normal to surface using vacuum ([#5d4e2fd](https://github.com/pol-sb/ATLAS/commit/5d4e2fd5bcc60f8c87ec3a0680dbcb5f7d8cc533))
- **dft**: added forgotten variable assignation ([#ead1680](https://github.com/pol-sb/ATLAS/commit/ead1680282a05cdf4ffd11ff87d6eff4f4abaedb))
- **dft**: added forgotten variable assignation when adding '.xyz' suffix in `run_dataframe_vasp_aiida_queue()` ([#3317616](https://github.com/pol-sb/ATLAS/commit/3317616df17530006e8000ac6635ab90be0a9b47))

### Misc

- **dft**: improved description of `generate_potential_mapping()` ([#26ff388](https://github.com/pol-sb/ATLAS/commit/26ff3881ca65e7de440327e542f6dadd3e77830a))
- **al_loop**: fixed typo on comments related to `gen_descriptors_and_concave_hull()` and `gen_descriptors()` ([#b0ec3a9](https://github.com/pol-sb/ATLAS/commit/b0ec3a93bc5a28d4c3f8ffcbb6c44f90e35cf662))
- **al_loop**: improved output message ([#4dd765f](https://github.com/pol-sb/ATLAS/commit/4dd765fecdbf025db56331477ccb4cb6d2daf0c9))
- **dft**: improve comments ([#bec3f2b](https://github.com/pol-sb/ATLAS/commit/bec3f2be0422ea2b18c940f8803cd4f1753116a5))
- **dft**: remove unnecessary check ([#1a2020f](https://github.com/pol-sb/ATLAS/commit/1a2020f850a4a94946689739881a71d9d8c25c03))
- **al_loop**: fixed typo ([#0cfc17f](https://github.com/pol-sb/ATLAS/commit/0cfc17f4df59f7c72b0911ca17560511476cd5e4))

## 0.20.10 (2025-02-03)

### Fix

- **core**: add github tags based version check ([#fb53c6d](https://github.com/pol-sb/ATLAS/commit/fb53c6d7c5df13e5e2e4dfcb938799245c179788))

## 0.20.9 (2025-02-02)

### Fix

- **al_loop**: add `atl_db_index` key to all structures before adding to db ([#a2063f0](https://github.com/pol-sb/ATLAS/commit/a2063f0294e4342755dc1866b4931bcd13c80a19))
- **al_loop**: add error for nan values inresult table for `TrainMACEModelCalculation` ([#41c06c2](https://github.com/pol-sb/ATLAS/commit/41c06c21a046f9b3bf17aaad1191defbec4dc86e))
- **core**: pinned `torch==2.4.1` to avoid breaking update to `torch==2.6` ([#7372413](https://github.com/pol-sb/ATLAS/commit/73724136fd94808e552188141e790b4df22612e8))
- **core**: fixes wrong path definition on `init_config_dir`. ([#2bf6a70](https://github.com/pol-sb/ATLAS/commit/2bf6a7038b6ca1eb1b9a554d79ad8c132d117022))
- **core**: fixes wrong path definition on `init_config_dir`. ([#1f51f1d](https://github.com/pol-sb/ATLAS/commit/1f51f1d165fa49c29e131b52463ba593ec02ad64))
- **core**: print used configuration path when running `atl_init_setup` after the first time ([#e0a57d3](https://github.com/pol-sb/ATLAS/commit/e0a57d3719e1bde455c258cc97ea6f7043baafd6))

### Misc

- **al_loop**: improve clarity of output when saving structures to database ([#68b6713](https://github.com/pol-sb/ATLAS/commit/68b67135622ef5003b5e5b5f0977f324eed3eebd))

## 0.20.8 (2025-01-31)

### Fix

- **dft**: fixed formatting error in get function `aiida_run_vasp_dft_... ([#4654448](https://github.com/pol-sb/ATLAS/commit/465444850ec085f33efc9d4ae0841676c63ad238))
- **dft**: fixed formatting error in get function `aiida_run_vasp_dft_database.py` ([#6f1d8e4](https://github.com/pol-sb/ATLAS/commit/6f1d8e4d68090f4ad3b98b9ddbcbb8d382bf9b83))
- **al_loop**: remove `ProcessMDSeedStructCalculation` temporary files after use ([#bb61df6](https://github.com/pol-sb/ATLAS/commit/bb61df6fd825b8129b66f65fcc42c978b95e36e0))
- **al_loop**: updated report generation to work with the on-the-fly active learning loop ([#0af22b6](https://github.com/pol-sb/ATLAS/commit/0af22b69287d2843349be2d8b2a8c8be22d05fe4))
- **al_loop**: catch `EOFError` exception when checking queue in `can_submit_calculation()` ([#fdf5208](https://github.com/pol-sb/ATLAS/commit/fdf5208757ecc1e159bd0f242dcbac32831c678b))

## 0.20.7 (2025-01-30)

### Fix

- **al_loop**: allow `ProcessMDSeedStructCalculation` to finish without an extrapolation plot. ([#af91762](https://github.com/pol-sb/ATLAS/commit/af9176264aaa4c10cbd763acdebc467ae607afca))
- **db_gen**: remove error when MP API key not found ([#ea8d318](https://github.com/pol-sb/ATLAS/commit/ea8d318f53fc467d38207c16c18d42e6a0415af8))
- **db_gen**: enable usage without MP API key ([#1f0da10](https://github.com/pol-sb/ATLAS/commit/1f0da1080fe5a73299c10a2d9e1b0ed85cfafd58))

### Misc

- **docs**: replaced 1.1/1.2 with Option A/B in `README.md` ([#849dd3d](https://github.com/pol-sb/ATLAS/commit/849dd3deb357733142b9f4d9d1b48ca5e846cee7))
- **docs**: added conda to installation instructions ([#a43aaf6](https://github.com/pol-sb/ATLAS/commit/a43aaf63f0a637139c5329926545948907f4dffd))
- **docs**: added conda to installation instructions ([#004b2b5](https://github.com/pol-sb/ATLAS/commit/004b2b55d9d8adf8c613306dd3b17e8d326979a7))
- **db_gen**: removed deprecated function `generate_surfaces_replacements()` ([#533521a](https://github.com/pol-sb/ATLAS/commit/533521a28bb7b1cb3d97288854432a17febd45aa))
- **docs**: replaced NNP with the more generic MLIP ([#33ab3e8](https://github.com/pol-sb/ATLAS/commit/33ab3e806fd74c881ec04cb0a35b56326a77b0d0))

## 0.20.6 (2025-01-30)

### Fix

- **al_loop**: `extrapolation_plot` now not required as an output now in `ProcessMDSeedStructCalculation`. This allows calculations that did not return extrapolating frames to finish without error codes. ([#6c665a2](https://github.com/pol-sb/ATLAS/commit/6c665a2221b377c6b1206c51d8c7d9d61fed2450))
- **al_loop**: added missing `atl_mace_eval_forces` key when converting structure to ase.Atoms for MD ([#94347c0](https://github.com/pol-sb/ATLAS/commit/94347c0ce811bf50136090fcbc603cd888527df2))
- **al_loop**: allowed gathering of DFT calculations from orm.CalcJob ([#47add66](https://github.com/pol-sb/ATLAS/commit/47add667bc76ebca7a7171a3a43dca234f9527ae))
- **al_loop**: added missing `spacegroup_kinds` key when converting structure to ase.Atoms for MD ([#de58e03](https://github.com/pol-sb/ATLAS/commit/de58e03d0fe8b7ecf3ac918b4d2546107452dd94))

### Misc

- **core**: pinned `mace-torch` dependency ([#7106d44](https://github.com/pol-sb/ATLAS/commit/7106d442caf6548e2af58a769811556f8a7219bb))
- **al_loop**: added missing documentations ([#f9d7af6](https://github.com/pol-sb/ATLAS/commit/f9d7af63fcbfe72a4aa1b0bde2b0056f498a2d39))

## 0.20.5 (2025-01-29)

### Misc

- **al_loop**: added documentation for container section ([#fc7708c](https://github.com/pol-sb/ATLAS/commit/fc7708c61e8db9f479192023c65b97e3b841b35c))

## 0.20.4 (2025-01-29)

### Fix

- **al_loop**: add handler to `atl_process_structure` for empty MD trajs ([#598ec1f](https://github.com/pol-sb/ATLAS/commit/598ec1f1e6bf74dbdbc5bd743e7caefa2647bc2f))
- **al_loop**: prepare paths in `atl_process_structure` for container use ([#1305373](https://github.com/pol-sb/ATLAS/commit/1305373860096564738f0027ce34b96f062fe68e))

## 0.20.3 (2025-01-28)

### Fix

- **al_loop**: fix incorrect latent space saving generating huge descriptor files ([#7a8c469](https://github.com/pol-sb/ATLAS/commit/7a8c469702f4d531ac3cc16a68aab787349214a7))
- **al_loop**: enable containerized code in simple active learning loop ([#664f697](https://github.com/pol-sb/ATLAS/commit/664f697278ba04cff3048b8a88167d3e49c1598e))

## 0.20.2 (2025-01-24)

### Fix

- **dft**: increased number of steps for IsolatedAtoms calculations ([#afbd09b](https://github.com/pol-sb/ATLAS/commit/afbd09bb3f850c7e5bf608a054de9c73e4130557))
- **al_loop**: made code compatible with MACE multihead replay fine-tuning ([#08944e2](https://github.com/pol-sb/ATLAS/commit/08944e2f7d63255ce7f6a52cccf13dd86469701c))
- **dft**: added IsolatedAtoms to initial database generation ([#96395ed](https://github.com/pol-sb/ATLAS/commit/96395ed190c8efcb00850e82c1521270929e7827))

### Misc

- **core**: updated documentation badge ([#79f34bf](https://github.com/pol-sb/ATLAS/commit/79f34bf7b40180b1226c9718058db8f066b3505f))

## 0.20.1 (2025-01-23)

### Fix

- **al_loop**: added TODO block with outlier filtering for VASP calculations ([#b592722](https://github.com/pol-sb/ATLAS/commit/b59272205594d5d7bb2cfec8663ffbcbf0d126d2))
- **al_loop**: added error when evaluating structures with missing element in MACE models ([#8c1d97a](https://github.com/pol-sb/ATLAS/commit/8c1d97a25a077c931cf2f9674be3aae711132a46))
- **al_loop**: improve error handling when mace evaluations fail ([#25bdb8c](https://github.com/pol-sb/ATLAS/commit/25bdb8c760ee91942ba3864c6ee2cd9dd19f4bc8))
- **core**: added default "unknown" version to version check in case it fails ([#df94bde](https://github.com/pol-sb/ATLAS/commit/df94bde589e5cc48cde8992df4a8c311c81cb82f))

### Misc

- **al_loop**: applied formatting and improved log messages ([#d53e86d](https://github.com/pol-sb/ATLAS/commit/d53e86d65545ffa227804f6998ce98d92416cb73))
- **core**: updated pre-commit config ([#9c87a53](https://github.com/pol-sb/ATLAS/commit/9c87a53e80f6ff9e414c70ee3552de17cb34e880))

## 0.20.0 (2025-01-20)

### Feature

- **al_loop**: add `atl_check_descr_combined.py` script for descriptor gathering and extrapolation check for simple active learning loop: - Introduced `atl_check_descr_combined.py`, a comprehensive script for checking extrapolation in the current iteration. - Implements descriptor generation using MACE and optional dimensionality reduction via an autoencoder. - Supports advanced extrapolation checks, including concave hull computation and visualization using `GMT.jl` and `ConcaveHull.jl`. ([#784ecdc](https://github.com/pol-sb/ATLAS/commit/784ecdc85a11691dbe426e3e90b9e4284ed930b7))
- **autoencoder**: Make autoencoder training compatible with on the fly active learning loop and provide improved path handling and error checks: ([#2b07fcb](https://github.com/pol-sb/ATLAS/commit/2b07fcb9f256bbd2d610203fe234faaa5d7d73ba))
- **al_loop**: Enhance MD structure processing in `atl_process_structure.py` ([#4f3a44b](https://github.com/pol-sb/ATLAS/commit/4f3a44b0939d2586ea72da93da790ff361e4ecae))
- **al_loop**: Add combined descriptor generation and concave hull analysis to simplified active learning loop ([#2943d0c](https://github.com/pol-sb/ATLAS/commit/2943d0cd681472581322d9f6c50939e2b8fffd2a))

### Fix

- **md**: add minimum number of atoms safeguard to `check_disconn_neighbors` md filter. ([#82caa11](https://github.com/pol-sb/ATLAS/commit/82caa11a679dcd38d39e31fab4dbc6d45aae181c))
- **core**: improve version check ([#6e94117](https://github.com/pol-sb/ATLAS/commit/6e941170f2589137c6cbc2a6dfcbb618b8988e7d))
- **al_loop**: Working ver. for `ProcessMDSeedStructCalculation` and `GetDescriptorsCombined` aling their parsers. ([#2947e88](https://github.com/pol-sb/ATLAS/commit/2947e889d9853da9835ebdbadadb7299b226c6ef))
- **docs**: fixed versions.html jinja template with missing elif statement ([#c53bfca](https://github.com/pol-sb/ATLAS/commit/c53bfca9fafd4d1f26b75677a392e55531e716c3))
- **docs**: updated README.md documentation links to master branch of documentation ([#9232175](https://github.com/pol-sb/ATLAS/commit/9232175caace654ed2634a5910d3157aba83d4f5))
- **dft**: fixed typo in `run_dataframe_vasp_aiida_queue` ([#e92dd82](https://github.com/pol-sb/ATLAS/commit/e92dd82e70f77ca963103ce6ab495d3a332d1cc7))

### Misc

- **docs**: update docs templates and config ([#2713b52](https://github.com/pol-sb/ATLAS/commit/2713b5264ba07c2cda68638dec16b26ef1ef9703))
- **docs**: add short commit hash to master/main branch in documentation ([#e541bfe](https://github.com/pol-sb/ATLAS/commit/e541bfee1ad73029a71419777af371d1b39a40ce))
- **dft**: added aiida-vasp parsing settings to `dft_settings.toml` ([#2f07c1c](https://github.com/pol-sb/ATLAS/commit/2f07c1c04040d85fce9886b825f136f6479e0e11))
- **core**: updated documentation.yml for gh-actions ([#1c63ed7](https://github.com/pol-sb/ATLAS/commit/1c63ed707e60050491902582d9ddadc2ceca04ed))
- **core**: updated documentation.yml for gh-actions ([#7ba647a](https://github.com/pol-sb/ATLAS/commit/7ba647a6b7e43fb1e5b4b331d336200353d0ed68))
- **core**: add documentation badge to README ([#631f5af](https://github.com/pol-sb/ATLAS/commit/631f5af320e69866aa1b33aa7fc41095f1efe502))
- **core**: updated documentation.yml for gh-actions ([#30b0c89](https://github.com/pol-sb/ATLAS/commit/30b0c89f26e2af57cafdca0c721fac338c4d5740))

## 0.19.12 (2024-12-28)

### Misc

- **core**: updated documentation.yml for gh-actions ([#2b1106b](https://github.com/pol-sb/ATLAS/commit/2b1106b7c34cb717367b06e9e3015e8d54745322))

## 0.19.11 (2024-12-28)

### Misc

- **core**: add versioning information to sphinx documentation ([#036379d](https://github.com/pol-sb/ATLAS/commit/036379dad1aedcb158d6258a6a5f456cf0db7708))

## 0.19.10 (2024-12-28)

### Fix

- **dft**: add `aiida_vasp.parser_settings` as input. Improve logging. ([#ff6f05a](https://github.com/pol-sb/ATLAS/commit/ff6f05a2b9fd4d2fa93f0774b8b4c5f4f73bdd0d))

### Misc

- **dft**: add structure count to  running calculations print statement ([#c63939d](https://github.com/pol-sb/ATLAS/commit/c63939dac44b644c13a6c084ba496d7cb44253b3))

## 0.19.9 (2024-12-23)

### Fix

- **core**: reformat `aiida_utils` and fix errors: - address indexing error when skipping completed calculations - create new structure and assign to list when updating in `update_db_with_dft_results` ([#4049eff](https://github.com/pol-sb/ATLAS/commit/4049effdbc861d27ded9037b207795446b76d575))
- **core**: add tags argument to  git fetch command for update checker ([#30b3d4f](https://github.com/pol-sb/ATLAS/commit/30b3d4fc219e9057685a1d03017e896d97500bba))

## 0.19.8 (2024-12-20)

### Fix

- **dft**: fix issue with phase collection ([#8dac04a](https://github.com/pol-sb/ATLAS/commit/8dac04a08a45b391552fc5199a829bb5e1095e10))

### Misc

- **dft**: reformatted `conversion.py` and updated call to extras on `get_struct_type()` ([#4434341](https://github.com/pol-sb/ATLAS/commit/44343413a799a6d52cea571a90a29c122ecc4796))

## 0.19.7 (2024-12-20)

### Fix

- **dft**: fixed unfinished loops and changed logging information during dft calculations ([#36262e3](https://github.com/pol-sb/ATLAS/commit/36262e36b0b0fa69134a916b16d04072078c9b24))

### Misc

- **core**: reenabled 'empty' custom print type ([#236570a](https://github.com/pol-sb/ATLAS/commit/236570a56181501556c84f1f4540f4f5a32d65c0))

## 0.19.6 (2024-12-19)

### Fix

- **dft**: enable usage of `calc_performed` to skip completed structures ([#28ec65e](https://github.com/pol-sb/ATLAS/commit/28ec65e4a5b62fb6edc3ff7df0822896b4be5379))

## 0.19.5 (2024-12-18)

### Fix

- **dft**: simplified calculation skipping logic in `update_db_with_dft_results()` and added try/except block to active_learning_utils ([#87267a3](https://github.com/pol-sb/ATLAS/commit/87267a3a78c29d817413503d309c686303030ab5))

## 0.19.4 (2024-12-17)

### Fix

- **dft**: changed logic for broken calculations in `update_db_with_dft_results()` ([#f717470](https://github.com/pol-sb/ATLAS/commit/f717470f51276b6082f2fd8ccad00afcd4cdc646))

## 0.19.3 (2024-12-17)

### Fix

- **dft**: changed `finished` with `finished_ok` in `update_db_with_dft_results` check ([#1128626](https://github.com/pol-sb/ATLAS/commit/1128626d7fecdeb0cf031a85efcd289220b28f7b))

## 0.19.2 (2024-12-16)

### Fix

- **dft**: added try block for different result gathering structures ([#dabc40f](https://github.com/pol-sb/ATLAS/commit/dabc40fdc67d4224b54eb5ddf5e32d3259298076))

## 0.19.1 (2024-12-14)

### Fix

- **dft**: added `update_db_with_dft_results` function to save DFT simulation results into an output file ([#f701b81](https://github.com/pol-sb/ATLAS/commit/f701b819f1b4064297b5c3c7237f07c5c0252708))

## 0.19.0 (2024-12-14)

### Feature

- **gen_db**: new adsorbate placement strategy (thank you @lll0606) and several improvements to dastabase generation ([#315a42b](https://github.com/pol-sb/ATLAS/commit/315a42b125cc6e4b01b3a17b8b091d0f5e26b518))
- **dft**: added `atl_run_dft_database` to entry points ([#5dfa2aa](https://github.com/pol-sb/ATLAS/commit/5dfa2aa1c004694f4652d5f399f1b918f86065bc))
- **dft**: introduced `atl_run_dft_database` utility to run dft calculations for mdb databases ([#c342594](https://github.com/pol-sb/ATLAS/commit/c3425944e872939d385c3ac6303c83a01b8bb3cd))

### Fix

- **core**: removed forced logger handler clearing from git tag gathering strategy ([#394a6b5](https://github.com/pol-sb/ATLAS/commit/394a6b52e2807af1fd1e5d437929dcc90f696f14))

### Misc

- **core**: updated clarity of several exceptions ([#610d435](https://github.com/pol-sb/ATLAS/commit/610d43579e545424f6f0436854899e8c4702ec0f))

## 0.18.3 (2024-12-04)

### Fix

- **al_loop**: reintroduced standard version of `TrainMACEModelCalculationParser()` ([#016279c](https://github.com/pol-sb/ATLAS/commit/016279c7f51a6ac6777347442203418c014e93b9))

## 0.18.2 (2024-12-02)

### Fix

- **al_loop**: add prepend_path to atl_process_structure to allow compatibility with docker ([#80a1289](https://github.com/pol-sb/ATLAS/commit/80a1289093b25f3a1e5e2db4499f6a0e9bdbd0ab))
- **al_loop**: change `get_concave_hull` aiida computer definition in `SimpleActiveLearningWorkChain` ([#1f391b6](https://github.com/pol-sb/ATLAS/commit/1f391b6283c428ab0e8c2c9bd5c2bc3c7d01711a))

## 0.18.1 (2024-11-28)

### Fix

- **al_loop**: moved juliacall deps. to `get_concave_hull_julia()` ([#9854aed](https://github.com/pol-sb/ATLAS/commit/9854aedeb84e165b462759c40e5eb91495a8ac21))
- **al_loop**: re-added wandb dependency for autoencoder training ([#ab9bddb](https://github.com/pol-sb/ATLAS/commit/ab9bddb4014def18aa1da63ab2e4c8131f9193e9))

### Misc

- **al_loop**: added message marking start of procedure to `atl_process_structure` script ([#04d617e](https://github.com/pol-sb/ATLAS/commit/04d617e85d2fb143becec52d2964ba7bafbd7f5f))

## 0.18.0 (2024-11-27)

### Feature

- **al_loop**: Included Maxwell-Boltzmann initialization of velocities for simple MD. ([#ffaa323](https://github.com/pol-sb/ATLAS/commit/ffaa3236eba82f9af36804d363c3fe539d1dea7f))

### Fix

- **initial_db**: commented julia imports from beginning of file to avoid errors on installs not using julia. ([#3111077](https://github.com/pol-sb/ATLAS/commit/3111077a6be9eb8c7f7731a60b76170d45ff9a86))
- **al_loop**: allow to generate plot for resumed calculations ([#477619e](https://github.com/pol-sb/ATLAS/commit/477619e3451404e9dc2b4da40ef72d1d117299dd))
- **al_loop**: improved active learning report plot ([#64e3107](https://github.com/pol-sb/ATLAS/commit/64e310768042010cdbf187a3211525f986c48bd8))
- **al_loop**: removed hardcoded value in active_learning_utils ([#19186fe](https://github.com/pol-sb/ATLAS/commit/19186feff4d9a1c7cf936942c0a79e3a26922a00))
- **al_loop**: modified descriptor get settings ([#00dc19e](https://github.com/pol-sb/ATLAS/commit/00dc19ef10ac28b400a52a906fb453d7a77c18d8))
- **al_loop**: added alternative E and F gathering in gather_md_E_F_data ([#78fcc91](https://github.com/pol-sb/ATLAS/commit/78fcc91b38079e5d709df603dcc8f1476b84ebb2))
- **al_loop**: improved `active_learning_utils.py` so they are more general ([#247ce3f](https://github.com/pol-sb/ATLAS/commit/247ce3f690ee55eed58e17b21dd220833f73e617))
- **al_loop**: replaced hardcoded settings with parameter dict for descriptor gathering ([#25c1433](https://github.com/pol-sb/ATLAS/commit/25c14339c0beab1765df0ea6c7d6aa49ec6fb885))
- **al_loop**: changed parameter gathering so that `atl_process_structure` works on non-gpu machines ([#fdccbc7](https://github.com/pol-sb/ATLAS/commit/fdccbc79c4bf8b1cba33617e95f6456843df1cf3))

### Misc

- **al_loop**: add inputs for enhanced error plot generation ([#2190828](https://github.com/pol-sb/ATLAS/commit/2190828a163dc0d0df8f86b8a6433dd0e5ff4cb9))
- **al_loop**: improve error plot generation ([#96699bf](https://github.com/pol-sb/ATLAS/commit/96699bf833f8b0880ebea0dc35a3c7f18e44c700))
- **al_loop**: move error plot to default al_loop report plot ([#0577b60](https://github.com/pol-sb/ATLAS/commit/0577b609ef17a1e90f26c2c3bba37baca6822112))
- **docs**: update README.md ([#6b26420](https://github.com/pol-sb/ATLAS/commit/6b26420067cc617686dba75957c67e7d099734bd))
- **docs**: update install docs, bump up python version and add julia ([#e09ccc8](https://github.com/pol-sb/ATLAS/commit/e09ccc81cccba2a295f74c6b26a8549549fdbdd7))
- **core**: refactored all python scripts using new ruff settings ([#4d010b2](https://github.com/pol-sb/ATLAS/commit/4d010b20fd31cf4d79ade6a66d9ed573577e6ebd))
- **core**: moved ruff settings to pyproject.toml and updated precommit configuration ([#655b2fc](https://github.com/pol-sb/ATLAS/commit/655b2fcde1218ab9f6229c8668e476868dbf3fac))

## 0.17.0 (2024-11-08)

### Feature

- **al_loop**: introduce first version of simplified active learning workchain ([#14e4332](https://github.com/pol-sb/ATLAS/commit/14e4332cb612715f102a8d7cd44a76fa1c7152c5))

## 0.16.6 (2024-11-06)

### Fix

- **al_loop**: fixed report logger printing double text ([#19859ab](https://github.com/pol-sb/ATLAS/commit/19859abaac8dbb5a8f610aa845afa1b49effa3ff))
- **al_loop**: improved report plot ([#cc28601](https://github.com/pol-sb/ATLAS/commit/cc286010823e38bda6ea72283dd48516a5c2737c))
- **al_loop**: define `num_cpus_large_struct` to use with large structures in md ([#429d79f](https://github.com/pol-sb/ATLAS/commit/429d79fa14964730266b890f91b8f758c32fb708))
- **al_loop**: fixed input ([#2211e2c](https://github.com/pol-sb/ATLAS/commit/2211e2cd86012fbbf914f8ac7e4403b0ab6329c4))
- **al_loop**: fixed input ([#b92194c](https://github.com/pol-sb/ATLAS/commit/b92194ce5cd68b190c9aec9ccd11493bc512af9b))

### Misc

- **docs**: updated admonition css ([#bf24ee2](https://github.com/pol-sb/ATLAS/commit/bf24ee26ab08c0cca0cbfe3a5043a164032392f1))
- **al_loop**: refactored code ([#bf4058d](https://github.com/pol-sb/ATLAS/commit/bf4058d37dfa0e61265749070a1e3d91dd4dd720))

## 0.16.5 (2024-11-05)

### Fix

- **al_loop**: added run call to resume cli command ([#29da12d](https://github.com/pol-sb/ATLAS/commit/29da12d6792e3e0ae04d05c83b47e4550efe9e43))
- **al_loop**: changed default value for `seed_min_num_structs` ([#5d270ae](https://github.com/pol-sb/ATLAS/commit/5d270aea21718b7b68dac1ac625ca2b2e70dcba4))

## 0.16.4 (2024-11-05)

### Fix

- **al_loop**: replaced `min_seed_frac` with `seed_min_num_structs` for md seeds ([#eda71f8](https://github.com/pol-sb/ATLAS/commit/eda71f87622179f34ad1f3fe1d2775d8b9d60e55))

## 0.16.3 (2024-11-05)

### Fix

- **al_loop**: mdb version now logged at the start of al_loop ([#d4cea8c](https://github.com/pol-sb/ATLAS/commit/d4cea8cd67a6d51c59a61d8716badad66966a536))
- **al_loop**: mdb version now logged at the start of al_loop ([#ec030db](https://github.com/pol-sb/ATLAS/commit/ec030dbb1b92346e6027206b0658d2c7734902c8))

## 0.16.2 (2024-11-05)

### Fix

- **al_loop**: added minimum md seed size ([#2464cf2](https://github.com/pol-sb/ATLAS/commit/2464cf28a082440421f83690f7c31a39ad5112a8))
- **al_loop**: added `can_do_advanced_extrapolation` check on iteration outline to allow for basic runs with no latent space computation ([#a9d4070](https://github.com/pol-sb/ATLAS/commit/a9d4070e3e9087ed17691f90efe42f8a26213c1a))

## 0.16.1 (2024-11-05)

### Fix

- **core**: changed git tag gathering strategy ([#bbe48bb](https://github.com/pol-sb/ATLAS/commit/bbe48bba58ec918a887c1e67715fe15b704ada4e))
- **al_loop**: add initial database size to al_report ([#b04f2c3](https://github.com/pol-sb/ATLAS/commit/b04f2c38d59dbc08bcb345c8a821e9e3513f24f2))
- **al_loop**: fixed conditional block preventing al_loop report generation ([#53373f2](https://github.com/pol-sb/ATLAS/commit/53373f29a3b4932560d8cd49987095c289844e98))

### Misc

- **docs**: update theme + add params for simple md ([#1c82597](https://github.com/pol-sb/ATLAS/commit/1c825973417502bfa1be175ddfbb9765511f3ce7))
- **docs**: add params for simple md ([#0667eb1](https://github.com/pol-sb/ATLAS/commit/0667eb1475f311cb60f70e58ab4984f34ee78d02))
- **docs**: modified `custom.css` ([#82ed9e8](https://github.com/pol-sb/ATLAS/commit/82ed9e879d70e5406084c9ab98197ab3dde94378))
- **docs**: modified `custom.css` ([#d80f6c1](https://github.com/pol-sb/ATLAS/commit/d80f6c1605d74996a8077f5eb43ebbfb741b2a5d))
- **docs**: update docs ([#2fe7908](https://github.com/pol-sb/ATLAS/commit/2fe7908cbd608e481648931d66c200e026e027bb))
- **docs**: update docs ([#78dc954](https://github.com/pol-sb/ATLAS/commit/78dc954d9400cec07a60f01a086d127f38591633))
- **docs**: moved calculation limiting info to `customization.md` ([#8682d2c](https://github.com/pol-sb/ATLAS/commit/8682d2cdf1dfcd31b7589dc471bbdc4396c49e4e))
- **docs**: updated CLI usage documentation ([#8594113](https://github.com/pol-sb/ATLAS/commit/859411335e3ad3cd8ec93e063e1e2b30271a0bab))
- **docs**: updated CLI usage documentation ([#3c5202d](https://github.com/pol-sb/ATLAS/commit/3c5202d2767e457a72e58fa7e636d772922e38f1))
- **docs**: updated CLI usage documentation ([#d04766b](https://github.com/pol-sb/ATLAS/commit/d04766b6e3479447b19520006982d7f8e680ea30))
- **docs**: add documentation for CLI usage ([#f626608](https://github.com/pol-sb/ATLAS/commit/f6266088046bd36e329d682c3fba52e6ebbc4d0e))

## 0.16.0 (2024-10-31)

### Feature

- **al_loop**: add resume function to al_loop ([#bc92d44](https://github.com/pol-sb/ATLAS/commit/bc92d44c77c58f1c5d75c54ca9242a569579a8e9))
- **al_loop**: add `copy_input_toml_file` function to al_loop ([#a4309e5](https://github.com/pol-sb/ATLAS/commit/a4309e5f723eeabe3686c185f7f9c8ffaaa45745))
- **al_loop**: add al_loop report subcommand ([#26d1a49](https://github.com/pol-sb/ATLAS/commit/26d1a49d62f82753d594176fbb94102dffc14da5))
- **al_loop**: add al_loop report subcommand ([#7def029](https://github.com/pol-sb/ATLAS/commit/7def02977922bd0c30ff02008a81254868a6b628))

### Fix

- **al_loop**: improved al_loop report representation ([#47fb827](https://github.com/pol-sb/ATLAS/commit/47fb827df14125434bafba7b4c9f4d9c895b1e2c))
- **core**: silenced git output when checking for updates ([#73548db](https://github.com/pol-sb/ATLAS/commit/73548db650c2537d23bd1084133061a17b613eac))

## 0.15.0 (2024-10-25)

### Feature

- **al_loop**: first iteration of the ASE-MD+descriptor+extrapolation PortableCode ([#28b4430](https://github.com/pol-sb/ATLAS/commit/28b4430208d62c6fa0013b3a161bd88335aeb9a1))

### Fix

- **al_loop**: slugify model names to remove unsupported characters ([#4fd825b](https://github.com/pol-sb/ATLAS/commit/4fd825bcfb4f06c1fd6d0a3e7519b1db17d393cb))

## 0.14.0 (2024-10-23)

### Feature

- **code**: add utilities for cache dir generation ([#2f5f9cb](https://github.com/pol-sb/ATLAS/commit/2f5f9cbf0d6c735b2887cd5ca5a4813a0f78c343))
- **gen_db**: use cache in surface generation ([#2279ae0](https://github.com/pol-sb/ATLAS/commit/2279ae00b4142bd70ceb450be10e85d899b4f47e))
- **phase**: introduce cache to structure generation ([#fe25d80](https://github.com/pol-sb/ATLAS/commit/fe25d80ea869b43a72ee0e9801ddf32c3dca87eb))
- **gen_db**: added parallel processing to `gen_surfaces_diff_miller` to speed up db generation ([#1a49a96](https://github.com/pol-sb/ATLAS/commit/1a49a9661255b4abe45509e123ac030308982954))

### Fix

- **autoencoder**: correctly assign device and dtype to tensors ([#2469f2d](https://github.com/pol-sb/ATLAS/commit/2469f2d3f346134705c1b228b12a0b58f6b98e4c))

### Misc

- **code**: update `cli_run_initial_config` to create cache folder ([#e6dc73d](https://github.com/pol-sb/ATLAS/commit/e6dc73d3af7c86ada7f149f699f1e8f2dab73e5f))
- **docs**: update cache docs ([#1ae899a](https://github.com/pol-sb/ATLAS/commit/1ae899a023dd80af290f868d74a2dd940814255f))
- **docs**: fixed typo ([#2e408f3](https://github.com/pol-sb/ATLAS/commit/2e408f3505b2d2ab01793a220f4e7e17e67fc1a9))
- **core**: moved `init_conf` to `code_utils.py` (ii) ([#7e034d7](https://github.com/pol-sb/ATLAS/commit/7e034d7b4c338c277158b96ff48ba6c26b30d7fa))
- **core**: moved `init_conf` to `code_utils.py` ([#0d6894c](https://github.com/pol-sb/ATLAS/commit/0d6894c46a1359042790f7b8414b217213705566))
- **core**: fixed dependency versions ([#1b88075](https://github.com/pol-sb/ATLAS/commit/1b8807548e47d115b6cbf753e01d9078e2e6fc7f))

## 0.13.0 (2024-10-21)

### Feature

- **gen_db**: improve handling of material prototypes and phase replacements ([#299ed8a](https://github.com/pol-sb/ATLAS/commit/299ed8a5b894606c62ec4f55a2db49ce2e13dea8))
- **autoencoder**: added option for dtype to autoencoder ([#0669b82](https://github.com/pol-sb/ATLAS/commit/0669b826aceeae28a658c96daaf1b960158d5b0c))

### Fix

- **gen_db**: keep base structures when limiting number of structures per phase ([#bc556db](https://github.com/pol-sb/ATLAS/commit/bc556db487143c389933a673bd6dcfc75c4ef7f8))
- **gen_db**: moved vacancy generation so it is applied to each phase ([#6271e8e](https://github.com/pol-sb/ATLAS/commit/6271e8e04c9a0cef04867b6523c5c9ca47b127ac))
- **al_loop**: updated inputs for  during loop ([#dbebeed](https://github.com/pol-sb/ATLAS/commit/dbebeed13ab7692387ea59038fdadef4825bcf97))
- **al_loop**: correctly gather computer in `generate_descriptors()` ([#410e94f](https://github.com/pol-sb/ATLAS/commit/410e94fe179065ecd5f2044407f0c0cb5bbe4e1c))
- **al_loop**: replaced deprecated `User.objects` with `User.collection` ([#e721d93](https://github.com/pol-sb/ATLAS/commit/e721d937a1e03051fbb0d45cd0d69452765eedc4))
- **al_loop**: added `atl_calc_limit` check before calcjob submission ([#e1f5017](https://github.com/pol-sb/ATLAS/commit/e1f501799614bb231c86d1ba74c0fb37f6a2afaa))
- **core**: removed git output from `get_last_tagged_version` ([#d74ff10](https://github.com/pol-sb/ATLAS/commit/d74ff10cbef1079f25b2fb8f8e3d498618d42384))

### Misc

- **core**: fix typo in `CHANGELOG.md` ([#0ee378e](https://github.com/pol-sb/ATLAS/commit/0ee378e1a3cb4dd678957a273ae7abcdc76b9726))
- **structure**: catch FutureWarning for new structure dataframe concatenation ([#b5796c6](https://github.com/pol-sb/ATLAS/commit/b5796c64af47d9e7f42f476bdcd1003dcb49e39d))
- **docs**: updated docs to include `atl_calc_limit` and improved phase explanation. ([#77121dc](https://github.com/pol-sb/ATLAS/commit/77121dc141389f9cbc9f5995b44d339386b13368))
- **docs**: updated docs ([#b2e4a73](https://github.com/pol-sb/ATLAS/commit/b2e4a73230c1da08df93e8c712a92d2f029c2286))
- **docs**: updated docs ([#cb5304f](https://github.com/pol-sb/ATLAS/commit/cb5304ff6f83c0aaff7ae147e404bc35e382b021))
- **docs**: updated docs ([#b0b2e7c](https://github.com/pol-sb/ATLAS/commit/b0b2e7c25a18cd46bada965e32553c2c471b7bd0))
- **docs**: added favicon ([#f865d30](https://github.com/pol-sb/ATLAS/commit/f865d30592ef32becc4e4343f46a5ad26621e117))
- **docs**: added favicon ([#83c015e](https://github.com/pol-sb/ATLAS/commit/83c015e5d193f6747186192d8a68d2cf300b6841))
- **core**: updated `InitialDatabase` and `Phase` **repr** methods ([#593cd76](https://github.com/pol-sb/ATLAS/commit/593cd76c916767187fe9dac72a9ee8f6b81444f4))
- **core**: marked conf.py as a version file ([#78c8b63](https://github.com/pol-sb/ATLAS/commit/78c8b6367cf670fd70f1d350c95a84e5aaeb7a5f))
- **docs**: updated documentation ([#2e300ba](https://github.com/pol-sb/ATLAS/commit/2e300ba157b6988620e3cafe502b150f158810fe))
- **docs**: updated docs ([#95760e1](https://github.com/pol-sb/ATLAS/commit/95760e1757a05a8484c6f2da876fc9f1c321b76a))
- **docs**: updated docs ([#ae57904](https://github.com/pol-sb/ATLAS/commit/ae57904e0f3fd6a3c0043a7a95a0ca475fbb265d))
- **docs**: updated docs ([#4b7c4e9](https://github.com/pol-sb/ATLAS/commit/4b7c4e93d703608936e1a3887e657eb1b895cbce))
- **docs**: updated docs ([#7284302](https://github.com/pol-sb/ATLAS/commit/728430279879fd5407cf056c6e988850903c4657))
- **docs**: updated `index.rst` ([#e29248f](https://github.com/pol-sb/ATLAS/commit/e29248fa94bdb2d32e5039390ed2a60dcd0fa2a6))
- **docs**: updated `README.md` ([#8513aa1](https://github.com/pol-sb/ATLAS/commit/8513aa1da2048079c06a85986a487edaccd76bad))
- **docs**: updated documentation ([#f2e2707](https://github.com/pol-sb/ATLAS/commit/f2e27071afb75983d00bb112698a594fd0cd65ca))
- **core**: added pip cache to github workflows ([#9082035](https://github.com/pol-sb/ATLAS/commit/9082035dfa88a919a912796f8fc62f2ea7c1ef2d))
- **core**: updated dependencies ([#c4ec121](https://github.com/pol-sb/ATLAS/commit/c4ec121064862109169595643e993c073ead4f3d))
- **docs**: updated workflows ([#37557f5](https://github.com/pol-sb/ATLAS/commit/37557f5903d35a5983eded5080333bdc753f54e4))
- **docs**: updated workflows ([#d7813a4](https://github.com/pol-sb/ATLAS/commit/d7813a411f756361b11384a225a8835ca4066253))
- **docs**: updated workflows ([#9fa22b3](https://github.com/pol-sb/ATLAS/commit/9fa22b373d74e869b1614e0d6f94f33b63fd4f10))
- **docs**: updated `README.md` ([#32f5e58](https://github.com/pol-sb/ATLAS/commit/32f5e588aa44794cf9472b1a2e4f8cc87c75e2e2))
- **docs**: added doc files ([#b47aca1](https://github.com/pol-sb/ATLAS/commit/b47aca140841c91cc9dd7fb4bed74168498f1cff))
- **docs**: added workflow file ([#bd052e4](https://github.com/pol-sb/ATLAS/commit/bd052e4f1547f3c924d5e62ae5de4601f469afe8))
- **docs**: added `.nojekyll` file ([#24ac3cc](https://github.com/pol-sb/ATLAS/commit/24ac3ccfef7e5efefa3ff62ddeed753cbfe29f87))
- **docs**: updated docs ([#97066e2](https://github.com/pol-sb/ATLAS/commit/97066e2ec2fd92c44fd8e308f50d6a8fccbbc438))
- **docs**: updated docs ([#9b5f688](https://github.com/pol-sb/ATLAS/commit/9b5f688055800271a3191f6c244209c729ed4749))
- **al_loop**: fixed logger output ([#d7e70af](https://github.com/pol-sb/ATLAS/commit/d7e70afbb26fbf9d68029ff9733e91d60ab2c497))
- **docs**: updated documentation ([#77c720e](https://github.com/pol-sb/ATLAS/commit/77c720ecf32dd7085269e00fa356dc1012c9c734))

## 0.12.0 (2024-10-15)

### Feature

- **gen_db**: added descriptor concave hull option to db_gen ([#01dbe76](https://github.com/pol-sb/ATLAS/commit/01dbe76814f3832aee1022647b9ff73f5d0ce7a6))

### Misc

- **core**: added `commitizen` to dev dependencies. ([#a8f2989](https://github.com/pol-sb/ATLAS/commit/a8f298934972d3b89f0e76b6608cbda927f53da6))

## 0.11.4 (2024-10-15)

### Fix

- **al_loop**: fixed `check_extrapolation_type` key being gathered from wrong dictionary ([#6386024](https://github.com/pol-sb/ATLAS/commit/6386024a1c2c7a496b733bfa475b84b9f2a4b87e))

### Misc

- **docs**: added input description for new extrapolation settings. ([#7794c68](https://github.com/pol-sb/ATLAS/commit/7794c68f2ab4e90ad4f64904b0bdd26d5a416fa6))
- **core**: added shapely dep. ([#05894e6](https://github.com/pol-sb/ATLAS/commit/05894e6d593155ac7b5ee1dcb86f662f295a7623))

## 0.11.3 (2024-10-14)

### Fix

- **gen_db**: perturbed structures now save material id ([#e859d7e](https://github.com/pol-sb/ATLAS/commit/e859d7e99ac0d6c09f1cd333e35e92634d73203c))

### Misc

- **gen_db**: added limit to number of phases in pie chart, remaining ones are grouped into `others`. ([#2fed433](https://github.com/pol-sb/ATLAS/commit/2fed433d60cf11af274bbc511c957d2c126270cd))
- **gen_db**: improved output and added deprecation warnings ([#2f3b778](https://github.com/pol-sb/ATLAS/commit/2f3b7786beb044905bb8934bb3f1fb77431e59eb))
- **structure**: simplified structure conversion ([#1575d91](https://github.com/pol-sb/ATLAS/commit/1575d9183765b9a9729f71210f43f727f07aa722))
- **structure**: updated structure representation ([#af9a608](https://github.com/pol-sb/ATLAS/commit/af9a60822878fc896bf3e87fb1490f50fd12f29c))
- **al_loop**: updated logger to rich logger ([#74ba05e](https://github.com/pol-sb/ATLAS/commit/74ba05ef9fc516cddf53e147624679d3e779cc27))
- **core**: improved report when missing ATL config file ([#a4abfdd](https://github.com/pol-sb/ATLAS/commit/a4abfddb7c01699ac1a37b94abe4eb6873e6c16a))

## 0.11.2 (2024-10-14)

### Fix

- **core**: added **version** directly to **init**.py ([#5429dd1](https://github.com/pol-sb/ATLAS/commit/5429dd163c00b34f8627754a6c7f941a99c47a94))

### Misc

- **docs**: updated formatting on CHANGELOG.md ([#7119a22](https://github.com/pol-sb/ATLAS/commit/7119a22193eb01716bbe02a1f3cf6ebd9ecc6ccf))

## 0.11.1 (2024-10-14)

### Fix

- **al_loop**: added check when submitting MACE energy evaluation to avoid empty lists being submitted. ([#c5744a7](https://github.com/pol-sb/ATLAS/commit/c5744a7f4bb275b3fbbde1dfa2b60a71d71769bc))
- **core**: fixed changelog format. ([#db3a77f](https://github.com/pol-sb/ATLAS/commit/db3a77f28ca64ab2439bc6d13799d1b0f4dcffe1))

### Misc

- **core**: updated changelog format. ([#4686f22](https://github.com/pol-sb/ATLAS/commit/4686f22aa43ab09bf62bbadd3f737a1880ca1ad1))

## 0.11.0 (2024-10-13)

### Feature

- **gen_db**: add bar chart and structure dict sorting. ([#9c7e277](https://github.com/pol-sb/ATLAS/commit/9c7e2770dfcd4348a502d14a3726b43e5714eea8))
- **al_loop**: added function to plot concave hull during the active learning loop ([#8948657](https://github.com/pol-sb/ATLAS/commit/8948657b402c541ed4d834c2b5fc0222d784ea9b))
- **al_loop**: add implementation of advanced extrapolation with concave hull and autoencoder support: ([#8ef5165](https://github.com/pol-sb/ATLAS/commit/8ef51659a52ce30952986f7914ad8e94ca0b7a88))

### Fix

- **gen_db**: use rcparams to update matplotlib rcparams globally. ([#675e683](https://github.com/pol-sb/ATLAS/commit/675e683b2b62616c619ebba2019e5acbf1818139))
- **gen_db**: allow runs without specifying optional paramaters. ([#6244b23](https://github.com/pol-sb/ATLAS/commit/6244b23f983957c87997ea9b38260fd6180df52a))
- **al_loop**: added parsing for `multiheads_finetuning` key for MACE training ([#50fd490](https://github.com/pol-sb/ATLAS/commit/50fd4902dd4f3e88bf2dd8f2e603a005f7f80a0b))

### Misc

- **gen_db**: rename central perturbation key. ([#b13304a](https://github.com/pol-sb/ATLAS/commit/b13304ae4069b22fad44db77c88f8eee02db041a))
- **gen_db**: added debug print ([#c26dd9e](https://github.com/pol-sb/ATLAS/commit/c26dd9e5da682dec93fc7c88704b263477eb4a24))
- **core**: pinned `mace-torch` library version ([#e40001e](https://github.com/pol-sb/ATLAS/commit/e40001ee1dae14db1a76aa2b0c2501f9234d7cae))

## 0.10.0 (2024-10-08)

### Feature

- **extrapolation**: add rng seed for autoencoder training ([#5f9df8c](https://github.com/pol-sb/ATLAS/commit/5f9df8cfd565c6a14d951baa01475f55b20e252a))
- **extrapolation**: include aiida entry points for autoencoder train and parsing ([#0868684](https://github.com/pol-sb/ATLAS/commit/0868684f14bbf79b5ee029adac28ae2cc9d29b36))
- **extrapolation**: added entry point for autoencoder training ([#7cbc062](https://github.com/pol-sb/ATLAS/commit/7cbc0629a71c9b7c7adef2dbbe27a57e071781bf))
- **db_gen**: added flag to avoid db overwriting, which adds a timestamp to the db_name ([#a6a61ee](https://github.com/pol-sb/ATLAS/commit/a6a61ee10c98d87d3f19f0db33bd4fa045d7c724))

### Misc

- **core**: changed logginghandler to `RichHandler` ([#98954f9](https://github.com/pol-sb/ATLAS/commit/98954f94dbf92ae2fb4d0aea1063557bdb5a9a0d))

## 0.9.1 (2024-10-08)

### Misc

- **core**: improved changelog generation ([#a99c322](https://github.com/pol-sb/ATLAS/commit/a99c3227a5e70a7c8fee4205a4365ee8907631e4))

## 0.9.0 (2024-10-08)

### Feature

- **db_gen**: add parameter for selecting the composition plot image format. ([#ce24695](https://github.com/pol-sb/ATLAS/commit/ce246956fae9c3a54541c70e9a98c0728ad3f9ef))

### Fix

- **gen_db**: Changed `overwrite_db` default to false ([#8978885](https://github.com/pol-sb/ATLAS/commit/89788855edbd0c66b918459ee5b28384a1566a9e))
- **structure**: Removed deprecated function ([#626001b](https://github.com/pol-sb/ATLAS/commit/626001b524fdb5b88fd1e729f677f92958f32d86))

### Misc

- **core**: added changelog template ([#d8f3215](https://github.com/pol-sb/ATLAS/commit/d8f32151e052d1dd6cdfca7e9f34a9b1ebf18a55))
- **gen_db**: improved database generation plot ([#17df15e](https://github.com/pol-sb/ATLAS/commit/17df15e91a81f9f4b013a3bb732cf4f8d43a0e5e))
- **core**: increased clarity of update warning ([#a26f3da](https://github.com/pol-sb/ATLAS/commit/a26f3da318bd5c039e9ccc38752750de585f22a7))
- **core**: updated CHANGELOG.md ([#b5f59d4](https://github.com/pol-sb/ATLAS/commit/b5f59d41f73baa6f6ceec3e5fc255a4295eb16b0))

## 0.8.0 (2024-10-07)

### Feature

- **db_gen**: added function to check for updates ([#b0e73ab](https://github.com/pol-sb/ATLAS/commit/b0e73abfc50e2755c11b7a1714d09ddb944d3925))

### Fix

- **db_gen**: fixed bug that prevented creation of 'central_atom_perturbation' modified structures ([#f31d6c4](https://github.com/pol-sb/ATLAS/commit/f31d6c4b3aaafc82649c31c6b7272f01df62abc6))
- **db_gen**: improved pie charts ([#d9d996d](https://github.com/pol-sb/ATLAS/commit/d9d996d1c6c48a23c6916f350375c1b856f39b5a))
- **db_gen**: refactored code to avoid circular imports ([#5c682a1](https://github.com/pol-sb/ATLAS/commit/5c682a1624459df6f58de3ca5d947baec82a08d6))

## 0.7.0 (2024-10-07)

### Feature

- **db_gen**: add parameter for disallowing database overwrite. ([#3a69386](https://github.com/pol-sb/ATLAS/commit/3a6938675c184a829a5bab907a86d8bb500d7472))
- **db_gen**: added pie charts to composition plots ([#594e3c6](https://github.com/pol-sb/ATLAS/commit/594e3c61831fef6149c1aee2ffc90a028526fa50))

### Fix

- **db_gen**: increased consistency with uuid and phase label storage. ([#7c38868](https://github.com/pol-sb/ATLAS/commit/7c3886853d09353aaa8a203bee57b22df90af69f))
- **db_gen**: increased consistency when saving structures with `targeted_modification`. ([#331263e](https://github.com/pol-sb/ATLAS/commit/331263efbd3e8e8ae1fa43eedf33aa107e8c612e))

## 0.6.11 (2024-10-06)

### Feature

- **structure**: added function to create mdb.Structure from InitialDatabase row. ([#98f6e22](https://github.com/pol-sb/ATLAS/commit/98f6e224ad47f161dfee8097502a9f8fbf5e4c6c))
- **gen_db**: added allow_modifications flag to config gile phase entries in order to avoid applying changes to desired base structures. ([#8543078](https://github.com/pol-sb/ATLAS/commit/85430787fd6111dc2639f6583b2f5e29db747c18))
- **gen_db**: added targeted modification `central_atom_octahedral` to database generation procedure ([#bdccd9f](https://github.com/pol-sb/ATLAS/commit/bdccd9f70bfdc9a5efe2008000872e73979a29d8))
- **database**: added attributes to Structure to identify targeted modification and al_loop_step. ([#46dbf80](https://github.com/pol-sb/ATLAS/commit/46dbf80307a332f57648548e29ceabc8aa2e233f))
- **al_loop**: add option to disable multiheads_finetuning for MACE 0.3.7 ([#9c69386](https://github.com/pol-sb/ATLAS/commit/9c69386f7cc126856be9428c929e9878e67a76e7))

### Fix

- **database**: modified database plot generation to avoid overlapping ([#e80fedc](https://github.com/pol-sb/ATLAS/commit/e80fedcd7fd220adc1a2f81cba7b0432e3166840))

## 0.6.10 (2024-10-03)

### BREAKING CHANGE

- The `system` key in the input TOML has been renamed to `database`, and users must now specify the path for the database and export options under this new key. ([#fbd88e2](https://github.com/pol-sb/ATLAS/commit/fbd88e23ee869691428d779da6ccd83da4dc2aed))
- `phase.name` is now slugified, which may affect downstream usage when accessing phase names. ([#a42ea14](https://github.com/pol-sb/ATLAS/commit/a42ea145324f855c75cf3a84ebc74435733fcc91))

### Feature

- **core**: improve surface generation with additional replacement and saving options ([#643ec8c](https://github.com/pol-sb/ATLAS/commit/643ec8c28b33811b60715f43f6e329d3910753f3))
- **core**: add displacement tracking and enhance structure generation ([#c63745d](https://github.com/pol-sb/ATLAS/commit/c63745d63f3cc5382c45159350710f90ec40e8c0))
- **core**: enhance structure handling and surface generation logic ([#fa6ddf3](https://github.com/pol-sb/ATLAS/commit/fa6ddf3cb8d4c1887c5f3c16b0f437404c1d9f5d))
- **core**: add phase structure limiting and improve phase handling ([#8a3e1a8](https://github.com/pol-sb/ATLAS/commit/8a3e1a8cbc7985d0568cb4a375d1954bc467b256))
- **core**: add element-specific vacancy generation and improve installation docs ([#08d5694](https://github.com/pol-sb/ATLAS/commit/08d5694cdbb897805a8775dfb06f96bd667913ad))
- **core**: add enhanced lattice perturbation and filtering options: ([#b5626fc](https://github.com/pol-sb/ATLAS/commit/b5626fcf58730a1e4be7ad84cee79609c86ad5f4))
  - Enhanced `perturb_min_displacement` to support structure filtering, phase selection, and random number generator seeding for lattice perturbations.
  - Introduced parameters `filters`, `only_use_base`, `use_phase`, and `rng_seed` to `perturb_min_displacement` for greater control over perturbation strategies.
  - Improved structure type conversion in vacancy generation and perturbation methods by using `to_bulk()`, `to_surface()`, and `to_cluster()` methods.
  - Updated string representations for `Structure` and `Phase` classes to improve readability of phase and structure details.
  - Added support for limiting the number of structures in lattice displacement operations.
  - Minor bug fixes and improvements to database and structure handling functions.    - `perturb_min_displacement` can now use different structures, although it still defaults to using base structures only unless specified otherwise to maintain compability. ([#b5626fc](https://github.com/pol-sb/ATLAS/commit/b5626fcf58730a1e4be7ad84cee79609c86ad5f4))
- **core**: add deprecation utilities and enhance warnings handling ([#37312dc](https://github.com/pol-sb/ATLAS/commit/37312dc2d1bd393be2c8789240978fbe7663e1b3))
- **database**: add support for random vacancy generation and structure enhancements ([#e04a3ac](https://github.com/pol-sb/ATLAS/commit/e04a3ac0a6f8d8d5476ea43c062497516b4843b4))
- **database**: enhance database generation with plotting, exporting, and structure replacement options ([#fbd88e2](https://github.com/pol-sb/ATLAS/commit/fbd88e23ee869691428d779da6ccd83da4dc2aed))
- **active_learning**: seed structure deletion is now optional in AL workflows ([#9617489](https://github.com/pol-sb/ATLAS/commit/96174895213ed41b87592c39ef40f2e344867ea9))
- **utils**: add support for adsorbate addition on surfaces in database ([#146727f](https://github.com/pol-sb/ATLAS/commit/146727fdfe10f2f842f58f99ee16f349a66341bd))
- **utils**: extend API key gathering to support environment variable ([#739e299](https://github.com/pol-sb/ATLAS/commit/739e299ee38f3f8b219f29e00195f1fe714f3038))
- **database**: enhance phase diagram and database initialization logic ([#a42ea14](https://github.com/pol-sb/ATLAS/commit/a42ea145324f855c75cf3a84ebc74435733fcc91))
- **phase_diagram**: upgrade phase diagram logic to support binary and ternary systems ([#21cc63e](https://github.com/pol-sb/ATLAS/commit/21cc63e57b08407fd087e6d09764e868e874a29f))
- **core**: added first test templates ([#9352880](https://github.com/pol-sb/ATLAS/commit/93528802017739106a59d2c2c3ba3959c3beb089))
- **gen_db**: add `apply_replacement_no_db()` ([#f20e6bf](https://github.com/pol-sb/ATLAS/commit/f20e6bf07cfad240ac43ec683cbf5a7089860bdf))
- **active_learning**: LAMMPS MD dump data now stored as compressed files ([#0c95f2c](https://github.com/pol-sb/ATLAS/commit/0c95f2cd5c766eaf69b1bfa0a880a707b835555f))
- **dashboard**: new logging option + rich cli output when running dashboard ([#75cdf58](https://github.com/pol-sb/ATLAS/commit/75cdf581a2253907512bd50bea3dc07d626effb6))
- **core**: added dev optional setup option ([#0ec4afe](https://github.com/pol-sb/ATLAS/commit/0ec4afe8c70f1cecdbcfb5b162dba03603402cae))
- **dashboard**: allow running dashboard in a separate process ([#a6d18ae](https://github.com/pol-sb/ATLAS/commit/a6d18ae9cbf7f09b74d43241b47b7353d123c1e4))
- **core**: added precommits ([#5eeea1e](https://github.com/pol-sb/ATLAS/commit/5eeea1e14cab762876bc7be1e57b6fbb666bb20f))
- **active_learning**: enhance active learning workflow with new md filters and descriptor settings ([#fe2da24](https://github.com/pol-sb/ATLAS/commit/fe2da243955248a6a6d22dd7ca062cbaeb0b1da5))

### Fix

- **gen_db**: fixed wrong argument for `gen_surfaces_diff_miller` ([#dc45ed8](https://github.com/pol-sb/ATLAS/commit/dc45ed8fa03359ed97c653dd74ebcd7969769635))
- **logs**: database report now showing in logs ([#52bf135](https://github.com/pol-sb/ATLAS/commit/52bf13564d0c9122e77809e0676c88de5bdba04c))
- **core**: fix phase name selection when filtering from db ([#6d9037e](https://github.com/pol-sb/ATLAS/commit/6d9037ee37bc27744af83dfeed5fc0022dd5efc7))
- **gen_db**: several fixes: ([#0240a46](https://github.com/pol-sb/ATLAS/commit/0240a4626156e1eb1d1937c0c4280d60202dcca8))
  - Reenabled ignore small structures during bulk generation.
  - Changed fixed seed with random generated one in case it is not given by the user.
  - Removed unused comments.
- **gen_db**: added conditional to avoid all types of perturbed structures being saved as cluster. ([#01fd2c0](https://github.com/pol-sb/ATLAS/commit/01fd2c0ec1a93c0a392df8d64c3ffd6e0742637f))
- **cli**: update toml handling for Python 3.11 compatibility ([#afc94d4](https://github.com/pol-sb/ATLAS/commit/afc94d4aa040030b6d43bc77b9c26079d457365a))
- **gen_db**: changed `fix_bottom_layers` message to warning. ([#631f420](https://github.com/pol-sb/ATLAS/commit/631f420928135411e86d7de187c3249ca935ef40))
- **gen_db**: fixed database generation functions: ([#f6e4466](https://github.com/pol-sb/ATLAS/commit/f6e44665f16b551ba78ccb3fe745872058a6c30a))
  - renamed `run_gen_initial_database` to `cli_run_gen_initial_database`
  - fixed import related to `cli_run_gen_initial_database`
  - converted numbers to float and int type after parsing database generation input .toml file
  - Added function for fixing bottom layers: `fix_bottom_layers`, however it is an unfinished which must be improved. It won't be used until finished.

- **core**: updated aiida-core dependency to avoid error #6519 ([#5be149b](https://github.com/pol-sb/ATLAS/commit/5be149bbc34c62deb17e5e5f94b11cb3414a2c83))
- **active_learning**: removed duplicate key ([#2ef361c](https://github.com/pol-sb/ATLAS/commit/2ef361c565e35bb1edec55741420009c4b68bdab))
- **active_learning**: fixed typo in send_calc_or_remove_structures()` ([#381a64a](https://github.com/pol-sb/ATLAS/commit/381a64a71d2ac0e2ecf902ffd6f48042bbdadf20))
- **active_learning**: added structure selection safeguard for `dft_structres` list ([#03f9779](https://github.com/pol-sb/ATLAS/commit/03f9779b3555aab84c5aaadbe2c8e05ee30c523a))
- **active_learning**: improved structure selection safeguard ([#c721582](https://github.com/pol-sb/ATLAS/commit/c721582b13f685189614ee950556868e0a67d2cd))
- **active_learning**: added safeguard check for filtered trajectories ([#e5ef27b](https://github.com/pol-sb/ATLAS/commit/e5ef27b5875cc3df5601f1e31c66cca3b8a99aca))
- **core**: updated imports on test files ([#46467f2](https://github.com/pol-sb/ATLAS/commit/46467f21ce014842488fa251d4d00dfc3cf48639))
- **active_learning**: only load MACE models on first iteration ([#e82744b](https://github.com/pol-sb/ATLAS/commit/e82744b1f93c46e1780544cf6cf0d2326e859a70))
- **active_learning**: changes to `send_calc_or_remove_structures` to avoid broadcasting errors: ([#295ff37](https://github.com/pol-sb/ATLAS/commit/295ff37fca18bb47f60ba3dc184e41fe552f2eae))
  - Added padding to extrapolating_frames array when size is different to error_f_structures or error_e_structures
  - Disabled shrinking from array masks to maintain array size.
- **active_learning**: added different padding strategies for energy and forces in `model_res_dict_to_arr()` ([#f5b0b21](https://github.com/pol-sb/ATLAS/commit/f5b0b219d93be56dcfa4070bb962f3dfaeccb56c))
- **active_learning**: updated `model_res_dict_to_arr` ([#624c1f9](https://github.com/pol-sb/ATLAS/commit/624c1f95749ec20a52c869101dca739b71d10dca))
- **dashboard**: Changed dashboard progressbar header ([#2c3e9b0](https://github.com/pol-sb/ATLAS/commit/2c3e9b0ce11713c0b711b14045337a56276a54d6))
- **dashboard**: Changed dashboard progressbar header ([#cc22899](https://github.com/pol-sb/ATLAS/commit/cc2289983bee20a85e3af1762fb5bd1f2a16dbf3))
- **active_learning**: added nan padding to uneven model_res_dict_to_arr arrays ([#a75c095](https://github.com/pol-sb/ATLAS/commit/a75c095c7e3d31a945cdcf6356ee19d293fb460c))
- **dashboard**: updated dashboard launch mechanism ([#a2ae5be](https://github.com/pol-sb/ATLAS/commit/a2ae5be51e9a909481e4a783aab50452e2b64b12))
- **active_learning**:mace model as absolute path (iv) ([#f703582](https://github.com/pol-sb/ATLAS/commit/f703582484e98afdd38a189d528a61544dcda4b7))
- **active_learning**:mace model as absolute path (iii) ([#ac43885](https://github.com/pol-sb/ATLAS/commit/ac4388551e8abd3d2e24e28c2f42f01e83c849fc))
- **active_learning**:mace model as absolute path (ii) ([#b6e104c](https://github.com/pol-sb/ATLAS/commit/b6e104c66328430558feac63bbc28a9f29909480))
- **gen_db**: changed incorrect import in cli_generate_init_db.py ([#7355f16](https://github.com/pol-sb/ATLAS/commit/7355f16f92941703f0efb6d7b10f6e9493143086))
- **core**: improved compatibility with tests ([#005f4b3](https://github.com/pol-sb/ATLAS/commit/005f4b3bedb9fab65af37e2c0866d6d150a9ca51))
- **active_learning**: resolve path on input files ([#df7a8b5](https://github.com/pol-sb/ATLAS/commit/df7a8b52c1b04c3e50dbf4f28c5f4a9f8ff708e6))
- **docs**: update documentation and modified wrong function call ([#69739d1](https://github.com/pol-sb/ATLAS/commit/69739d1ee2d524ca0e735d29ee8e369c5516c3fa))
- **dashboard**: update dashboard counter ([#1afc7d7](https://github.com/pol-sb/ATLAS/commit/1afc7d7f5b34c3557210abe70777fceb029ed5e7))
- **active_learning**: allow TrainMACEModelCalculationParser to gather any model type ([#d3d55b0](https://github.com/pol-sb/ATLAS/commit/d3d55b01ca07bbac56fc571359f8eacaf0873d13))
- **docs**: improved README.md writing ([#af6d94d](https://github.com/pol-sb/ATLAS/commit/af6d94d02fe76ac89a3953e21a11a0bcf4bccc16))
- **core**: applied uniform formatting to most files in the library ([#daa62dd](https://github.com/pol-sb/ATLAS/commit/daa62dd4c09a311965aa0d74f9b88a00e540aecf))
- **dashboard**: improve iteration display and change progressbar iteration ([#e03b413](https://github.com/pol-sb/ATLAS/commit/e03b41331515c56bb9de9422fc7414180fcb25f3))
- **active_learning**: remove filtered md structures from db and fix descriptor executable ([#13f7b83](https://github.com/pol-sb/ATLAS/commit/13f7b83074967988e93e3c51a7b9709af965ea41))
- **active_learning**: add cwd in remote as path for PortableCodes to run ([#e0c3101](https://github.com/pol-sb/ATLAS/commit/e0c3101c4273830a10cbe4349ab096587221af2f))
- **active_learning**: changed forces unit ([#6ffd858](https://github.com/pol-sb/ATLAS/commit/6ffd8581a867124dab7bd13d77eb1220e5e7fbbc))
- **active_learning**: allow to use MACE_forces key on extxyz files ([#8d36397](https://github.com/pol-sb/ATLAS/commit/8d36397c0a059c7fdbd20048bce12decf52f8e5f))
- **core**: allow to use functions from `aiida_utils` without using aiida ([#61bfad6](https://github.com/pol-sb/ATLAS/commit/61bfad6f7be6ea07dbdadfcbce4d99af661ceeb2))

### Refactor

- **core**: replace deprecated structure conversion with new methods ([#f12aa96](https://github.com/pol-sb/ATLAS/commit/f12aa96420401968bd7a7509c5f03f764062ae52))
- **core**: reduced line length on rich cli outputs ([#2f59483](https://github.com/pol-sb/ATLAS/commit/2f5948334ce1dcfd81710a89fd863899b0839aec))
- **core**: renamed cli files ([#c5e4e73](https://github.com/pol-sb/ATLAS/commit/c5e4e7358b449859dddab63cb046733ffc028de7))
- **core**: separated command line scripts into individual files ([#2df2e93](https://github.com/pol-sb/ATLAS/commit/2df2e9368325d16fef4d96dc55fdf4113336a2ab))

## v0.3.9 (2024-06-19)

development version
