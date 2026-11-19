# Third-party notices

The root [Apache License 2.0](LICENSE) covers project-authored application code and documentation, except material identified under separate terms. It does not relicense dependency code, datasets, care text or fine-tuned weights. See the project's [NOTICE](NOTICE).

## G2 ConvNeXt-Tiny + BiGRU

The current Windows bundle uses the project-trained G2 state dict. See the [G2 model card](MODEL_CARD.md), the [upstream ConvNeXt license](desktop/notices/convnext/LICENSE), and the [binary component notices](desktop/notices/NOTICE.md). The model card records the PlantSeg v7 training subset and the dataset terms noted in the training records.

## Historical DINOv2

DINOv2-base by Meta AI / FAIR (Meta Platforms, Inc. and affiliates), [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base/tree/f9e44c814b77203eaa57a6bdbbd535f21ede1415), revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

Copyright (c) Meta Platforms, Inc. and affiliates. Upstream portions are licensed under the Apache License, Version 2.0; the [full license](desktop/notices/dinov2/LICENSE) is retained. The [pinned model card](https://huggingface.co/facebook/dinov2-base/raw/f9e44c814b77203eaa57a6bdbbd535f21ede1415/README.md) declares `apache-2.0`. The Hugging Face team authored that card; research/model authorship belongs to the upstream team.

Project modifications: fine-tuning the final four transformer blocks and final layer normalization, adding a nine-class classifier, selecting epoch 6 from 224px training, and using that same checkpoint for 280px inference/evaluation with horizontal-flip logit averaging. No retraining was performed for this licensing update. The project owner requested distribution of the [fine-tuned checkpoint](https://github.com/VanHuynh205/TomatoAI-Public/releases/tag/v0.1.0-dinov2-v1-280) through GitHub Release assets for collaboration, while training-data clearance remains unresolved. This notice does not grant rights in training images or declare a license for the fine-tuned checkpoint. Meta/FAIR and the research authors do not endorse Tomato Desktop.

Research reference: Maxime Oquab; Timothée Darcet; Théo Moutakanni; Huy Vo; Marc Szafraniec; Vasil Khalidov; Pierre Fernandez; Daniel Haziza; Francisco Massa; Alaaeldin El-Nouby; Mahmoud Assran; Nicolas Ballas; Wojciech Galuba; Russell Howes; Po-Yao Huang; Shang-Wen Li; Ishan Misra; Michael Rabbat; Vasu Sharma; Gabriel Synnaeve; Hu Xu; Hervé Jegou; Julien Mairal; Patrick Labatut; Armand Joulin; Piotr Bojanowski. *DINOv2: Learning Robust Visual Features without Supervision*. 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193).

## Training and evaluation datasets

[DATASET_ATTRIBUTIONS.md](DATASET_ATTRIBUTIONS.md) records the seven historical DINOv2 fine-tuning source families, exact available versions/revisions, publisher and paper author names, DOI/source links, license links, project modifications and unresolved image-rights evidence. Rachin is test-only; Taiwan Tomato and TOM2024 are external comparisons, not fine-tuning sources.

No dataset images, archives or derivative images are included in this source release. CC terms and any third-party image rights remain separate from the code license. Attribution does not itself establish permission to publish the current fine-tuned weights.

## Care content

Eight approved Vietnamese offline-reference cards are adapted from Plantwise/CABI factsheets. They and their drafts retain [CC BY-SA 4.0 with source-specific attribution](infra/knowledge/CONTENT-LICENSE.md), including ©CAB International, named authors, original URLs and change notices. See the [source map](docs/care/licensed-source-map.md) and [review scope](docs/care/review.md). Source PDFs, photos and logos are not distributed. The project approval does not imply expert verification or CABI endorsement.

## Frontend components

The bundled frontend includes React 19.3.1, React DOM 19.3.1 and Scheduler 0.28.0 under their MIT licenses. Copyright (c) Meta Platforms, Inc. and affiliates. Keep their original notices:

- [React license](desktop/notices/frontend/react-19.3.1/LICENSE)
- [React DOM license](desktop/notices/frontend/react-dom-19.3.1/LICENSE)
- [Scheduler license](desktop/notices/frontend/scheduler-0.28.0/LICENSE)
- [Component provenance](desktop/notices/frontend/components.json)

The current frontend also includes GSAP and Lenis. The component inventory records the versions and original license files for all bundled frontend components.

## Python runtime and Microsoft components

Runtime/build dependencies are pinned in [the Windows CPU lock](desktop/requirements-runtime-cpu.lock). The packaging specification collects installed distribution metadata and complete license trees, alongside these notices; dependency licenses are not replaced by the project's license. A source checkout alone is not a standalone executable distribution. The Windows Release ZIP retains the licenses and notices applicable to its collected dependencies; consult its release manifest and verification report for the tested scope.

The pywebview WebView2 binding includes a separately attributed [Microsoft WebView2 SDK notice](desktop/notices/microsoft-webview2-sdk/LICENSE.txt) and [provenance](desktop/notices/microsoft-webview2-sdk/provenance.json). The Edge WebView2 Evergreen Runtime is a separately installed Microsoft prerequisite; it is not bundled or relicensed here.

## Historical project license

Earlier public project revisions were released under MIT. Their [original copyright and permission notice](desktop/notices/project-MIT-legacy.txt) is preserved. The Apache-2.0 transition does not revoke rights already granted for those revisions.
