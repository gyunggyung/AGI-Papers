<h1 align="center"> 🌟  AGI-Papers 🌟 </h1>

<p align="center">
  </a> 
    </a>
  <em>
    LLM
    · NLP
  </em>
  <br />
  <em>
    Text2All
    · All2All
  </em>
  <br />
  <em>
    Multi-modal
    · Multi-task
  </em>
  <br />

</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="licenses" src="https://img.shields.io/github/license/gyunggyung/LLM-Papers?style=flat-square"></a>
  <a href="https://github.com/gyunggyung/LLM-Papers/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/gyunggyung/LLM-Papers?style=flat-square&color=yellow"></a>
  <a href="https://github.com/gyunggyung/LLM-Papers/blob/master/watchers">
    <img alt="GitHub watching" src="https://img.shields.io/github/watchers/gyunggyung/LLM-Papers?style=flat-square&color=ff69b4"></a>
  <a href="https://github.com/gyunggyung/LLM-Papers/graphs/contributors">
    <img alt="contributors" src="https://img.shields.io/badge/contributors-welcome-yellowgreen?style=flat-square"></a>
</p>

<div align="center">
    <sub> Let's find out the latest and various LLM-related papers. 🙇‍♂️🙇‍♀️ by <a href="https://github.com/gyunggyung/LLM-Papers/stargazers">Stargazers</a>  </sub>
</div>

# AGI
- [🦍 Gorilla: Large Language Model Connected with Massive APIs](https://gorilla.cs.berkeley.edu/)
> Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, their potential to effectively use tools via API calls remains unfulfilled. This is a challenging task even for today's state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call. We release Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls. When combined with a document retriever, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly. To evaluate the model's ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs. The model and code of Gorilla are available at https://github.com/ShishirPatil/gorilla.

- [÷🦎 Chameleon: Plug-and-Play Compositional Reasoning with GPT-4](https://github.com/lupantech/chameleon-llm)
> Large language models (LLMs) have achieved remarkable progress in various natural language processing tasks with emergent abilities. However, they face inherent limitations, such as an inability to access up-to-date information, utilize external tools, or perform precise mathematical reasoning. In this paper, we introduce Chameleon, a plug-and-play compositional reasoning framework that augments LLMs to help address these challenges. Chameleon synthesizes programs to compose various tools, including LLM models, off-the-shelf vision models, web search engines, Python functions, and rule-based modules tailored to user interests. Built on top of an LLM as a natural language planner, Chameleon infers the appropriate sequence of tools to compose and execute in order to generate a final response. We showcase the adaptability and effectiveness of Chameleon on two tasks: ScienceQA and TabMWP. Notably, Chameleon with GPT-4 achieves an 86.54% accuracy on ScienceQA, significantly improving upon the best published few-shot model by 11.37%; using GPT-4 as the underlying LLM, Chameleon achieves a 17.8% increase over the state-of-the-art model, leading to a 98.78% overall accuracy on TabMWP. Further studies suggest that using GPT-4 as a planner exhibits more consistent and rational tool selection and is able to infer potential constraints given the instructions, compared to other LLMs like ChatGPT.

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
> Believable proxies of human behavior can empower interactive applications ranging from immersive environments to rehearsal spaces for interpersonal communication to prototyping tools. In this paper, we introduce generative agents--computational software agents that simulate believable human behavior. Generative agents wake up, cook breakfast, and head to work; artists paint, while authors write; they form opinions, notice each other, and initiate conversations; they remember and reflect on days past as they plan the next day. To enable generative agents, we describe an architecture that extends a large language model to store a complete record of the agent's experiences using natural language, synthesize those memories over time into higher-level reflections, and retrieve them dynamically to plan behavior. We instantiate generative agents to populate an interactive sandbox environment inspired by The Sims, where end users can interact with a small town of twenty five agents using natural language. In an evaluation, these generative agents produce believable individual and emergent social behaviors: for example, starting with only a single user-specified notion that one agent wants to throw a Valentine's Day party, the agents autonomously spread invitations to the party over the next two days, make new acquaintances, ask each other out on dates to the party, and coordinate to show up for the party together at the right time. We demonstrate through ablation that the components of our agent architecture--observation, planning, and reflection--each contribute critically to the believability of agent behavior. By fusing large language models with computational, interactive agents, this work introduces architectural and interaction patterns for enabling believable simulations of human behavior.
- [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366)
> Recent advancements in decision-making large language model (LLM) agents have demonstrated impressive performance across various benchmarks. However, these state-of-the-art approaches typically necessitate internal model fine-tuning, external model fine-tuning, or policy optimization over a defined state space. Implementing these methods can prove challenging due to the scarcity of high-quality training data or the lack of well-defined state space. Moreover, these agents do not possess certain qualities inherent to human decision-making processes, specifically the ability to learn from mistakes. Self-reflection allows humans to efficiently solve novel problems through a process of trial and error. Building on recent research, we propose Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities. To achieve full automation, we introduce a straightforward yet effective heuristic that enables the agent to pinpoint hallucination instances, avoid repetition in action sequences, and, in some environments, construct an internal memory map of the given environment. To assess our approach, we evaluate the agent's ability to complete decision-making tasks in AlfWorld environments and knowledge-intensive, search-based question-and-answer tasks in HotPotQA environments. We observe success rates of 97% and 51%, respectively, and provide a discussion on the emergent property of self-reflection.

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
> Like people, LLMs do not always generate the best text for a given generation problem on their first try (e.g., summaries, answers, explanations). Just as people then refine their text, we introduce SELF-REFINE, a framework for similarly improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an output using an LLM, then allow the same model to provide multi-aspect feedback for its own output; finally, the same model refines its previously generated output given its own feedback. Unlike earlier work, our iterative refinement framework does not require supervised training data or reinforcement learning, and works with a single LLM. We experiment with 7 diverse tasks, ranging from review rewriting to math reasoning, demonstrating that our approach outperforms direct generation. In all tasks, outputs generated with SELF-REFINE are preferred by humans and by automated metrics over those generated directly with GPT-3.5 and GPT-4, improving on average by absolute 20% across tasks.

- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580)
> Solving complicated AI tasks with different domains and modalities is a key step toward advanced artificial intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower this. Based on this philosophy, we present HuggingGPT, a framework that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Hugging Face, HuggingGPT is able to cover numerous sophisticated AI tasks in different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards advanced artificial intelligence.
- [Auto-GPT: An Autonomous GPT-4 Experiment](https://github.com/Significant-Gravitas/Auto-GPT)
> Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

- [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)
> There is a rapidly growing number of large language models (LLMs) that users can query for a fee. We review the cost associated with querying popular LLM APIs, e.g. GPT-4, ChatGPT, J1-Jumbo, and find that these models have heterogeneous pricing structures, with fees that can differ by two orders of magnitude. In particular, using LLMs on large collections of queries and text can be expensive. Motivated by this, we outline and discuss three types of strategies that users can exploit to reduce the inference cost associated with using LLMs: 1) prompt adaptation, 2) LLM approximation, and 3) LLM cascade. As an example, we propose FrugalGPT, a simple yet flexible instantiation of LLM cascade which learns which combinations of LLMs to use for different queries in order to reduce cost and improve accuracy. Our experiments show that FrugalGPT can match the performance of the best individual LLM (e.g. GPT-4) with up to 98% cost reduction or improve the accuracy over GPT-4 by 4% with the same cost. The ideas and findings presented here lay a foundation for using LLMs sustainably and efficiently.

- [LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://leandojo.org/)
> Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection— a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): the first LLM-based prover that is augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 96,962 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.



- [Can Language Models Teach Weaker Agents? Teacher Explanations Improve Students via Theory of Mind]()
> Large Language Models (LLMs) perform complex reasoning by generating explanations for their predictions. However, a complementary goal of explanations is to also communicate useful knowledge that improves weaker agents. Hence, we investigate whether LLMs also make good teachers for weaker agents. In particular, we consider a student-teacher framework between two LLM agents and study if, when, and how the teacher should intervene with natural language explanations to improve the student's performance. Since communication is expensive, we define a budget such that the teacher only communicates explanations for a fraction of the data, after which the student should perform well on its own. We decompose the teaching problem along four axes: (1) if teacher's test time intervention improve student predictions, (2) when it is worth explaining a data point, (3) how the teacher should personalize explanations to better teach the student, and (4) if teacher explanations also improve student performance on future unexplained data. We first show that teacher LLMs can indeed intervene on student reasoning to improve their performance. Next, we propose a Theory of Mind approach, in which the teacher builds two few-shot mental models of the student. The first model defines an Intervention Function that simulates the utility of an intervention, allowing the teacher to intervene when this utility is the highest and improving student performance at lower budgets. The second model enables the teacher to personalize explanations for a particular student and outperform unpersonalized teachers. We also demonstrate that in multi-turn interactions, teacher explanations generalize and learning from explained data improves student performance on future unexplained data. Finally, we also verify that misaligned teachers can lower student performance to random chance by intentionally misleading them.

- [Kosmos-2: Grounding Multimodal Large Language Models to the World]
> We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at this https URL.


- [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403)
> We introduce PaLM 2, a new state-of-the-art language model that has better multilingual and reasoning capabilities and is more compute-efficient than its predecessor PaLM. PaLM 2 is a Transformer-based model trained using a mixture of objectives. Through extensive evaluations on English and multilingual language, and reasoning tasks, we demonstrate that PaLM 2 has significantly improved quality on downstream tasks across different model sizes, while simultaneously exhibiting faster and more efficient inference compared to PaLM. This improved efficiency enables broader deployment while also allowing the model to respond faster, for a more natural pace of interaction. PaLM 2 demonstrates robust reasoning capabilities exemplified by large improvements over PaLM on BIG-Bench and other reasoning tasks. PaLM 2 exhibits stable performance on a suite of responsible AI evaluations, and enables inference-time control over toxicity without additional overhead or impact on other capabilities. Overall, PaLM 2 achieves state-of-the-art performance across a diverse set of tasks and capabilities.

- [MotionGPT: Finetuned LLMs are General-Purpose Motion Generators](https://arxiv.org/abs/2306.10900)
> Generating realistic human motion from given action descriptions has experienced significant advancements because of the emerging requirement of digital humans. While recent works have achieved impressive results in generating motion directly from textual action descriptions, they often support only a single modality of the control signal, which limits their application in the real digital human industry. This paper presents a Motion General-Purpose generaTor (MotionGPT) that can use multimodal control signals, e.g., text and single-frame poses, for generating consecutive human motions by treating multimodal signals as special input tokens in large language models (LLMs). Specifically, we first quantize multimodal control signals into discrete codes and then formulate them in a unified prompt instruction to ask the LLMs to generate the motion answer. Our MotionGPT demonstrates a unified human motion generation model with multimodal control signals by tuning a mere 0.4% of LLM parameters. To the best of our knowledge, MotionGPT is the first method to generate human motion by multimodal control signals, which we hope can shed light on this new direction. Codes shall be released upon acceptance.



- [Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks](https://arxiv.org/pdf/2305.14201.pdf)
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)
- [Model Card and Evaluations for Claude Models](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf?fbclid=IwAR2OBPueSK8lT2Xq2JibUSybkI6K3Y3Mw8asj6brKakF3O7IsZh_oQ-x-A0)

- [Augmenting Language Models with Long-Term Memory](https://huggingface.co/papers/2306.07174)
- [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302)
- [Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/pdf/2306.02707.pdf)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Dr. LLaMA: Improving Small Language Models Through Generative Data Augmentation](https://github.com/zguo0525/Dr.llama)

- [The FLAN Instruction Tuning Repository](https://github.com/google-research/FLAN)
- [Phoenix: Democratizing ChatGPT across Languages](https://arxiv.org/abs/2304.10453)

- [RedPajama-INCITE](https://together.ai/blog/redpajama-models-v1?fbclid=IwAR2CAjHzWIZaHVZF14XNVG8DLmBmmxQp9Gm3ZYiVqDUGvTPEF3_O6D5RZX4)

- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

- [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)
- [Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291.pdf)
- [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/pdf/2302.14045v1.pdf)
- [Tackling multiple tasks with a single visual language model](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf%EC%97%90%EC%84%9C)
- [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846)

- [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602.pdf)
- [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/pdf/2109.01247.pdf)

- [∞-former: Infinite Memory Transformer](https://arxiv.org/abs/2109.00301)
- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)
- [Augmented Language Models: a Survey](https://arxiv.org/pdf/2302.07842.pdf)

- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597v1.pdf)
- [Structure and Content-Guided Video Synthesis with Diffusion Models](https://arxiv.org/pdf/2302.03011.pdf)
- [MusicLM: Generating Music From Text](https://arxiv.org/pdf/2301.11325.pdf)

- [InstructGPT : Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining](https://arxiv.org/abs/2210.10341)

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Provable Copyright Protection for Generative Models](https://arxiv.org/abs/2302.10870)
- [What learning algorithm is in-context learning? Investigations with linear models](https://arxiv.org/abs/2211.15661)
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [PAL: Program-aided Language Models](https://arxiv.org/pdf/2211.10435.pdf)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761.pdf)

- [LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/file/1574548786327032/LLaMA--Open-and-Efficient-Foundation-Language-Models.pdf)
- [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)

- [LLaMA-based ChatGPT training](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama), [ChatLLaMA](https://github.com/juncongmoo/chatllama)
- [RLHF: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2204.05862.pdf)
- [BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores](https://dl.acm.org/doi/pdf/10.1145/3503221.3508417)

- [LLaMA-7B](https://huggingface.co/spaces/chansung/LLaMA-7B), [LLAMA Up-data](https://github.com/hunkim/llama-up-data), [LLaMA: INT8 edition](https://github.com/tloen/llama-int8), [UForm](https://github.com/unum-cloud/uform)

- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198), [Blog](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)
- [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)

- [How to use UForm](https://colab.research.google.com/drive/1-8-kyfaN6bExVCmQmEbO7YWxpj)
- [How to create KoChatLLaMA](https://docs.google.com/document/d/1-Kj4s_gP90X8__gOPU)

- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)
- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)
- [GPU and learning method required for KoChatLlaMA fine-tuning]()
- [Advantages and Problems of UForm]()

- [GPT-4 is coming next week – and it will be multimodal, says Microsoft Germany](https://www.heise.de/news/GPT-4-is-coming-next-week-and-it-will-be-multimodal-says-Microsoft-Germany-7540972.html)
- [MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation](https://arxiv.org/abs/2303.00628)
- [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages](https://arxiv.org/abs/2303.01037)
- [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io/#demo)


### Paper to read
- [Tightly-Integrated Generative Encoder-Decoder Representation]()
- [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://arxiv.org/abs/2303.04671)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks](https://arxiv.org/abs/2302.13939)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning](https://github.com/cloneofsimo/lora)

- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)
- [FLAN: Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652.pdf)
- [T0: Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207.pdf)
- [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688.pdf)
- [The Wisdom of Hindsight Makes Language Models Better Instruction Followers](https://arxiv.org/abs/2302.05206.pdf)
- [Exploring the Benefits of Training Expert Language Models over Instruction Tuning](https://paperswithcode.com/paper/exploring-the-benefits-of-training-expert.pdf)
- [Unsupervised Imputation of Non-ignorably Missing Data Using Importance-Weighted Autoencoders](https://arxiv.org/abs/2101.07357)
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073.pdf)
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741.pdf)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)
- [Large Language Models with Controllable Working Memory](https://arxiv.org/abs/2211.05110.pdf)
- [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247.pdf)
- [Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/pdf/2301.00704v1.pdf)
- [Structure and Content-Guided Video Synthesis with Diffusion Models](https://arxiv.org/abs/2302.03011.pdf)
- [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
- [A hunt for the Snark: Annotator Diversity in Data Practices](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/c5dbc2c146b7443447d43a344b4a22a359b32b16.pdf)
- [Accurate global machine learning force fields for molecules with hundreds of atoms](https://www.science.org/doi/full/10.1126/sciadv.adf0873)
- [Algorithms with More Granular Differential Privacy Guarantees](https://arxiv.org/pdf/2209.04053.pdf)
- [Anomaly Clustering: Grouping Images into Coherent Clusters of Anomaly Types](https://research.google/pubs/pub51881/)
- [Are we cobblers without shoes? Making Computer Science data FAIR](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/962979aa7d369692aa1919dcb517a5e5a3d0fa66.pdf)
- [Code Generation for In-Place Stencils](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/26cdf8c85c428bbff96ff6e7a07fe313e9844d68.pdf)
- [Creating, Calibrating, and Validating Large-Scale Microscopic Traffic Simulation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/0e0284222cf7d2d690f1b277b970de8e3f61294a.pdf)
- [Increasing Impact of Mobile Health Programs: SAHELI for Maternal and Child Care](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/4e67c21de7542a2eb61edd37ece44c86928e00b1.pdf)
- [Designing Responsible AI: Adaptations of UX Practice to Meet Responsible AI Challenges](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/4045f0e0e61d89b6b1eacad4b861e86631d5e660.pdf)
- [Developer Productivity for Humans: A Human-Centered Approach to Developer Productivity](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9994260)
- [Development of a Machine Learning Model for Sonographic Assessment of Gestational Age](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2800007)
- [Drug Design on Quantum Computers](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/20b69430c8fd64f170d5c7ba0a0535aa51e70ab6.pdf)
- [Estimates of broadband upwelling irradiance from GOES-16 ABI](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/146e5813afdf4f8ba02d18a801c6ed06de27d7d5.pdf)
- [Information Processing and Management](https://reader.elsevier.com/reader/sd/pii/S0306457322003508?token=D58C3F3F1C12E28D51A5515FA2F3608D25FB798EBDEE89D440A941BFCFAC8872866F56B9D4B863D69728CCCB26963764&originRegion=us-east-1&originCreation=20230216172152)
- [Flake Aware Culprit Finding](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/a422ac172dbb3c6521f2bc6c83d363695f4e911c.pdf)
- [Flexible Budgets in Restless Bandits: A Primal-Dual Algorithm for Efficient Budget Allocation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/00074e5dacad1fe2e7f03f65274993fcfbc860bd.pdf)
- [Helpful Neighbors: Leveraging Neighbors in Geographic Feature Pronunciation](https://watermark.silverchair.com/tacl_a_00535.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAr0wggK5BgkqhkiG9w0BBwagggKqMIICpgIBADCCAp8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMGaBmgqnePMpUEr_1AgEQgIICcHADbjfmiT9rEwbAOxm_3_0F7rZ2w_ETcJECYlEEpAHZUqPFKMP6eae59zDGkfZVaF3pq7w2JGFl-PJqQyT_fdQxIadVrVhW-xsibjoGYmAnciIFFIedkpSKJqXP2k4KG-lrRck9cEprUnptSHqq_VHWZFCETv1b9EuexXPfjHGEYDVCFEHER2nfwGzp4E7iMMvlOm53M3gU_WUmXcJOZcAnOgmYCyNh01R7f3vzI4Y1zKz9lhGNTL86mU-wFv-8aUWWbtIFLoiCgGaAgmp3m5N-vmefhqL4UtutrkAfTwq0xxg6VmEoSpBk52mIBusp07qxFA8s0wXFu3mBkJLJAVFqLtBJsyl2jic59TmHY4BDt7An6WcozDInnxILNbClDQAqVIh6OLNOJ4r9MXFUAyTrX3lbqN52SlygwV4K3z_ZFdGM7ndaWuJg5Ou2UdXfnaFrSjyvcNRutcUaupT7DGcXV3L7wb7b-XKA4OoAdSS8x1s-hhNW4l-I_iuiTx5vRAsVY0EyhOR2EPajiLKiEVN7SDEZ9ihAtef312FBsqu9xK10m14chdHdxvuIrwtuBwcVaj1rMPIzRPLU_l3Hxfa2x735pDjHZhDxzm2rEyXtlOXY-VIiv2gXD539gB1KFQ3VCjVnvOsesXlMue9a0LPPU6Tyb7wpo43ERNs8U5s5BRmomjOsUMyeM_Zp47O94WNyMo6V_F2bjhEZJq-lJZxtyqGcN6QFXiSdGoMIG5xQasrVUButsE3tVJELEjZU_j5rRKt8iXYPvRXsCNHZWG0_ezt_NsHM6H9xgz6vTFtezIGiLmpYZIBcVcU51Iq5bw)
- [High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/ca9caaef5f92cb1c1088897a9654a79ccdcb2bd4.pdf)
- [Helpful Neighbors: Leveraging Neighbors in Geographic Feature Pronunciation](https://watermark.silverchair.com/tacl_a_00535.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAr0wggK5BgkqhkiG9w0BBwagggKqMIICpgIBADCCAp8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM9k6JNPdG5DftjzF0AgEQgIICcAmosEi5f1ji5KSTr9mz9ZwOEdbRU73_HgVTuXYzZyu500yymmCBEL9V1rg0Dv6ynhoWfTZJIozQ8Qd30uTpLwf5N167hksw3DLEAIsOMG4zoYFts1uazpEn42BVoQANpM-xj9ruJM5YQuYWscLrXzmCGAI93pQSxPlkjiwPQtsUlX713I5Qn_QDaV8_sGTMu0yw_Dxbz9p5dtFy_wovYDhzCB4W7yQlEaVhFUvk04A3PEfVmIi2c0sctYptcz8QDEnPI-kGpD_mplO65b0QHmYYBWriZWc13d4M6RyUWAgrpMQinmZtFRSxYjBof-BoqnlLEMpxfMECrFRTK3Jx3sKfWPpEfye_VpcJ4b-PqDpCHuMnjSL2jqmcRN5Fp9vhyjECs7HMepK618KC8NAJRN4_YbQnyOm-UT5MzaL72aQsaChkr1xZi9JOUe1xvqdjazEH9y9VgvW7cM8XCZZVLum87s2zsx8qzjqPMsfZLRoc8HsB9Tu-dvuxGPQK3gKetGRQjatjbKn4a3CSWhAij5ayG66Aa4AaYi-33iqG4EcBH6WLQG9JouFiojC1Pkj03O0evnvXP1zy11CZ_3w9QjScEfXdUh3no0dukyw5pnsrlxDWdGhZZFQx1NfGcYDOi6luzEpAijwKIVm7FogdhVg_hD63JoddNI9y7iTESYBvkBwt249qyhnpJ2Z5osFuipKGyOEYog0DYh1jxY-BIu1CY-vb9uLr1Jud7zCXkIwxfYswmLXiaJJQQo3v6Rle5Wek3MxP4CN4X4mwtQrrFJFKwddfMa6Ki5ViOTS-ckW_zLoNaORYu8UKfz_2fPjxNg)
- [Infrastructuring Care: How Trans and Non-Binary People Meet Health and Well-Being Needs through Technology](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/98030d5f96d6f383c90e40fcbf79425919f178d3.pdf)
- [KwikBucks: Correlation Clustering with Cheap-Weak and Expensive-Strong Signals](https://openreview.net/pdf?id=p0JSSa1AuV)
- [Learning to Bid in Contextual First Price Auctions](https://arxiv.org/pdf/2109.03173.pdf)
- [Machine Learning for Healthcare: A Bibliometric Study of Contributions from Africa](https://www.preprints.org/manuscript/202302.0010/v1)
- [Scalable Decision-Focused Learning in Restless Multi-Armed Bandits with Application to Maternal and Child Health](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/c394143cc1cb501f710a304f15983e70ed22fa8a.pdf)
- [Robust Planning over Restless Groups: Engagement Interventions for a Large-Scale Maternal Telehealth Program](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/c3043326e08d5775bc766ab494254e02560cfbe0.pdf)
- [Recitation-Augmented Language Models](https://arxiv.org/pdf/2210.01296.pdf)
- [RL4ReAl: Reinforcement Learning for Register Allocation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/cfad891e4eee1a4c38f55bff5117b3ba7aa65392.pdf)
- [Quantum Simulation of Exact Electron Dynamics can be more Efficient than Classical Mean-Field Methods](https://research.google/pubs/pub52012/)
- [Quantum simulation of exact electron dynamics can be more efficient than classical mean-field methods](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/231543ccbbbd24d725563e79c01743fbb4eb06e0.pdf)
- [Propeller: A Profile Guided, Relinking Optimizer for Warehouse-Scale Applications](https://dl.acm.org/doi/10.1145/3575693.3575727)


- [Deepmind: Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)
- [Deepmind: Building safer dialogue agents](https://www.deepmind.com/blog/building-safer-dialogue-agents)
- [Deepmind: Competitive programming with AlphaCode](https://www.deepmind.com/blog/competitive-programming-with-alphacode)
- [Deepmind: Mastering Stratego, the classic game of imperfect information](https://www.deepmind.com/blog/mastering-stratego-the-classic-game-of-imperfect-information)
- [Deepmind: DeepMind’s latest research at NeurIPS 2022](https://www.deepmind.com/blog/deepminds-latest-research-at-neurips-2022)
- [Deepmind: Building interactive agents in video game worlds](https://www.deepmind.com/blog/building-interactive-agents-in-video-game-worlds)
- [Deepmind: Discovering novel algorithms with AlphaTensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor)
- [Deepmind: AlphaFold reveals the structure of the protein universe](https://www.deepmind.com/blog/alphafold-reveals-the-structure-of-the-protein-universe)
- [Deepmind: Exploring the beauty of pure mathematics in novel ways](https://www.deepmind.com/blog/exploring-the-beauty-of-pure-mathematics-in-novel-ways)
- [Deepmind: Nowcasting the next hour of rain](https://www.deepmind.com/blog/nowcasting-the-next-hour-of-rain)
- [Deepmind: Putting the power of AlphaFold into the world’s hands](https://www.deepmind.com/blog/putting-the-power-of-alphafold-into-the-worlds-hands)
- [Google Research: Deciphering clinical abbreviations with privacy protecting ML](https://ai.googleblog.com/2023/01/deciphering-clinical-abbreviations-with.html)
- [Google Research: Google Research, 2022 & beyond: Language, vision and generative models](https://ai.googleblog.com/2023/01/google-research-2022-beyond-language.html)
- [Google Research: Google Research, 2022 & beyond: Responsible AI](https://ai.googleblog.com/2023/01/google-research-2022-beyond-responsible.html)
- [Google Research: Learning with queried hints](https://ai.googleblog.com/2023/01/learning-with-queried-hints.html)
- [Google Research: Open Source Vizier: Towards reliable and flexible hyperparameter and blackbox optimization](https://ai.googleblog.com/2023/02/open-source-vizier-towards-reliable-and.html)
- [Google Research: Google Research, 2022 & beyond: ML & computer systems](https://ai.googleblog.com/2023/02/google-research-2022-beyond-ml-computer.html)
- [Google Research: Real-time tracking of wildfire boundaries using satellite imagery](https://ai.googleblog.com/2023/02/real-time-tracking-of-wildfire.html)
- [Google Research: Breaching the 2 LMP Approximation Barrier for Facility Location with Applications to k-Median](https://research.google/pubs/pub51938/)
- [Google Research: Chimane-Mosetén](https://research.google/pubs/pub52097/)
- [Google Research: Differentially Private All-Pairs Shortest Path Distances: Improved Algorithms and Lower Bounds](https://research.google/pubs/pub51926/)
- [Google Research: Differentially Private Fair Division](https://research.google/pubs/pub51931/)
- [Google Research: DiffQG: Generating Questions on Paired Sentences](https://research.google/pubs/pub52078/)
- [Google Research: Assessment of Security Defense of Native Programs Against Software Faults](https://link.springer.com/chapter/10.1007/978-3-031-02063-6_5)
- [Google Research: Adaptive mixing of auxiliary losses in supervised learning](https://research.google/pubs/pub51874/)
- [OpenAI: Multimodal Neurons in Artificial Neural Networks](https://openai.com/blog/multimodal-neurons/)
- [OpenAI: DALL·E: Creating Images from Text](https://openai.com/blog/dall-e/)
- [OpenAI: CLIP: Connecting Text and Images](https://openai.com/blog/clip/)
- [OpenAI: Image GPT](https://openai.com/blog/image-gpt/)
- [OpenAI: Jukebox](https://openai.com/blog/jukebox/)
- [OpenAI: Solving Rubik’s Cube with a Robot Hand](https://openai.com/blog/solving-rubiks-cube/)
- [OpenAI: Multimodal Neurons in Artificial Neural Networks](https://openai.com/blog/multimodal-neurons/)
- [OpenAI: CLIP: Connecting Text and Images](https://openai.com/blog/clip/)
- [OpenAI: Image GPT](https://openai.com/blog/image-gpt/)
- [OpenAI: MuseNet](https://openai.com/blog/musenet/)
- [OpenAI: Emergent Tool Use from Multi-Agent Interaction](https://openai.com/blog/emergent-tool-use/)


## before 2023

- [2013/01] **[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)**    

- [2014/12] **[Dependency-Based Word Embeddings](https://www.aclweb.org/anthology/P14-2050.pdf)**

- [2015/07] **[Neural Machine Translation of Rare Words with Subword Units](https://www.aclweb.org/anthology/P16-1162.pdf)**

- [2014/07] **[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)** : *GloVe*

- [2016/06] **[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/pdf/1606.04640.pdf)** : *Siamese CBOW*

- [2016/07] **[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)** : *fastText*

- [2014/09] **[Sequence to Sequence Learningwith Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)** : *seq2seq*

- [2017/07] **[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)** : *Transformer*

- [2017/08] **[Learned in Translation: Contextualized Word Vectors](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)** : *CoVe*

- [2018/01] **[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)** : *ULMFIT*

- [2018/02] **[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)** : *ELMo* 

- [2018/06] **[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)** : *GPT-1* 

- [2018/10] **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)** : *BERT*     

- [2019/02] **[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** : *GPT-2* 

- [2019/04] **[Language Models with Transformers](https://arxiv.org/abs/1904.09408)** 

- [2019/08] **[Neural Text Generation with Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf)** 

- [2019/01] **[Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291.pdf)** *XLM* 

- [2019/01] **[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/pdf/1901.11504.pdf)** : *MT-DNN*    

- [2019/01] **[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)** : *Transformer-XL*    

- [2019/06] **[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)** : *XLNet*

- [2019/04] **[The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf)**

- [2019/09] **[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)** 

- [2019/01] **[BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf)** : *BioBERT* 

- [2019/03] **[SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676.pdf)** : *SciBERT*

- [2019/04] **[ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342.pdf)** : *ClinicalBERT* 

- [2019/06] **[HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/pdf/1905.06566.pdf)** : *HIBERT* 

- [2019/07] **[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)** : *SpanBERT*

- [2019/04] **[Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323.pdf)** 

- [2019/08] **[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf)** 

- [2019/07] **[Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://arxiv.org/pdf/1907.11932.pdf)**


- [2019/07] **[R-Transformer: Recurrent Neural Network Enhanced Transformer](https://arxiv.org/abs/1907.05572)** : *R-Transformer*


- [2019/09] **[FREELB: ENHANCED ADVERSARIAL TRAINING FOR LANGUAGE UNDERSTANDING](https://arxiv.org/pdf/1909.11764.pdf)** : *FREELB*

- [2019/09] **[Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks](https://arxiv.org/pdf/1909.11515.pdf)**


- [2019/10] **[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)** : *T5*



- [2018/07] **[Subword-level Word Vector Representations for Korean](https://www.aclweb.org/anthology/P18-1226.pdf)**



- [2019/08] **[Zero-shot Word Sense Disambiguation using Sense Definition Embeddings](https://malllabiisc.github.io/publications/papers/EWISE_ACL19.pdf)**

- [2019/06] **[Bridging the Gap between Training and Inference for Neural Machine Translation](https://arxiv.org/pdf/1906.02448.pdf)**

- [2019/06] **[Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts](https://arxiv.org/pdf/1906.01267.pdf)**

- [2019/07] **[A Simple Theoretical Model of Importance for Summarization](https://www.aclweb.org/anthology/P19-1101.pdf)**

- [2019/05] **[Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)**

- [2019/07] **[We need to talk about standard splits](http://wellformedness.com/papers/gorman-bedrick-2019.pdf)**

- [2019/07] **[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412v1.pdf)** : *ERNIE 2.0*

- [2019/05] **[SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/pdf/1905.00537.pdf)** : *SuperGLUE*

- [2020/01] **[Towards a Human-like Open-Domain Chatbot](https://arxiv.org/pdf/2001.09977.pdf)** + *[Google AI Blog](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html)*

- [2020/03] **[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555.pdf)** : *ELECTRA*

- [2019/04] **[Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324.pdf)** : *Mask-Predict*

- [2020/01] **[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451.pdf)** : *Reformer*

- [2020/04] **[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150.pdf)** : *Longformer*

- [2019/11] **[DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536.pdf)** : *DialoGPT*

- [2020/01] **[Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977.pdf)**

- [2020/04] **[You Impress Me: Dialogue Generation via Mutual Persona Perception](https://arxiv.org/abs/2004.05388.pdf)**

- [2020/04] **[Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637.pdf)**

- [2020/04] **[ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues](https://arxiv.org/abs/2004.06871.pdf)**
: *ToD-BERT*

- [2020/04] **[SOLOIST: Few-shot Task-Oriented Dialog with A Single Pre-trained Auto-regressive Model](https://arxiv.org/abs/2005.05298.pdf)** : *SOLOIST*

- [2020/05] **[A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/abs/2005.00796.pdf)**

- [2019/07] **[ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation](https://arxiv.org/abs/1907.05339.pdf)** : *ReCoSa*

- [2020/04] **[FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://arxiv.org/abs/2004.02178)** : *FastBERT*

- [2020/01] **[PoWER-BERT: Accelerating BERT inference for Classification Tasks](https://arxiv.org/abs/2001.08950)** : *PoWER-BERT*

- [2019/10] **[DistillBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)** : *DistillBERT*

- [2019/10] **[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)** : *TinyBERT*

- [2019/11] **[Not Enough Data? Deep Learning to the Rescue!](https://arxiv.org/abs/1911.03118)**

- [2018/12] **[Conditional BERT Contextual Augmentation](https://arxiv.org/abs/1812.06705)**

- [2020/03] **[Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)**

- [2020/04] **[FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/abs/2004.11795)** : *FLAT*

- [2019/12] **[Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)** : *BiT*

- [2019/04] **[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)** : *ERNIE*

- [2019/07] **[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)** : *ERNIE 2.0*

- [2020/06] **[ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph](https://arxiv.org/abs/2006.16934)** : *ERNIE-ViL*

- [2020/12] **[ERNIE-Doc: A Retrospective Long-Document Modeling Transformer](https://arxiv.org/abs/2012.15688)** : *ERNIE-Doc*

- [2021/07] **[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)** : *ERNIE 3.0*

- [2022/10] **[Beyond English-Centric Bitexts for Better Multilingual Language Representation Learning](https://arxiv.org/abs/2210.14867)**

- [2017/03] **[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)**

- [2020/10] **[DiPair: Fast and Accurate Distillation for Trillion-Scale Text Matching and Pair Modeling](https://arxiv.org/abs/2010.03099)** : *DiPair*

- [2021/08] **[Distilling Transformers for Neural Cross-Domain Search](https://arxiv.org/abs/2108.03322)**

- [2020/06] **[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)** : *DeBERTa*

- [2020/11] **[VEGA: Towards an End-to-End Configurable AutoML Pipeline](https://arxiv.org/abs/2011.01507)** : *VEGA*

- [2020/12] **[FILTER: An Enhanced Fusion Method for Cross-lingual Language Understanding](https://arxiv.org/abs/2009.05166)** : *FILTER*

- [2019/12] **[StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577)** : *StructBERT*

- [2019/04] **[Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding](https://arxiv.org/abs/1904.09482)** : *MT-DNN*

- [2021/05] **[Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation](https://arxiv.org/abs/2105.08919)**

# MLLMArxivTalk
최신 MLLM 관련 스터디. 기본 오후에 진행. 논문, 강의, 코드, 뉴스, 블로그 등 다양한 자료로 학습.

MLLM, LLM, NLG, Dialogue, Reinforcement learning, Distillation, Efficient, Sentence similarity, multiple tasks, multimodal, Stable diffusion, TTS, Text-To-Video, All-To-All, 우주, 생명, 지능, 윤리, 규제, 법, 노화, 의학, 투자, 개발, 인프라, 디자인, 경영, ETC... 

유망 스타트업 C레벨, 국내외 탑티어 연구자, 국내외 탑티어 대학, 대학원 재학생과 졸업생, 석학, 교수 등 **A급 인재들이 최신 논문, 강의 등 스터디 및 프로젝트 진행.**

기본 매주 수요일 오후 7시반. 사전 학습 없이 논문 읽기 최대 20분, 토론 최대 40분. 한 번에 1 ~ 10개 논문, 강의 등 진행. 지금까지는 항상 3개. 주제 논문 선정은 자유. 탑티어 학회 논문 및 프로젝트 제작 예정.

주말을 포함하여, 거의 매일 추가 스터디 존재. 흥미로운 주제거나 참여 되는 날만 중간에 들어와서 중간에 나가도 무관. 모든 규칙은 협의 가능. 오프라인 모임도 예정. 자율 참여.


## 스터디 규칙
1. 영어만 사용은 금지. 한국어 중심 사용. 특수 용어는 영어 사용.
2. 1주일에 논문 2개 이상 스터디. 되는 사람은 10개 이상.
3. 3분에서 20분 현장에서 논문 읽기. 5분에서 30분 토론.
4. 1시간 스터디 시, 바로 나가도 됨. 원할 때 10분 이하 참여도 무관. 자유롭게 진행. 2시간 매일도 가능.
5. 각자 더 뛰어난 게 있다는 것을 인지. 다들 대단한 분들이니 질문 많이 하고, 정보 공유 자주.
6. 본인이 하기로 한 일만은 수행. 한다고 말하고, 안 하는 것은 민폐다.
7. 기본적으로 녹화 후 내부 공유.
8. 정보를 혼자 알게 쓰지 말고, 다 같이 알게 말하기.
9. 개인 사정으로 스터디 탈퇴 시, 자기소개에 인사 작성.
10. 여러 기관 좋은 규칙 붙여넣기.
11. 팀에 도움이 된다고 판단하면, 위 규칙을 모두 무시하고 행동.
12. 추가.

## Basic knowledge

| mathematics              | machine learning    | Transformer                            | Hugging Face     |
|:------------------------------------------------------------------------------:|:------------:|:------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![](image/ml.PNG)](https://mml-book.github.io/) | [![](image/pr.PNG)](http://users.isr.ist.utl.pt/~wurmd/Livros/school/) | [![](https://www.perlego.com/_next/image?url=https%3A%2F%2Fwww.perlego.com%2Fbooks%2FRM_Books%2Fpackt_pub_vpnckweg%2F9781838826239_500_750.jpg&w=1440&q=75)]() | [![image](https://user-images.githubusercontent.com/26733242/226218579-eb9fb6d8-ad50-4424-bb44-5f42cc9e3be3.png)](https://transformersbook.com/) |
| **[mathematics for machine learning](https://mml-book.github.io/)** | **[Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/)** | **[Getting Started with Google BERT](https://mml-book.github.io/)** | **[Natural Language Processing with Transformers](https://transformersbook.com/)** |

