# PromptSub - Data

## Directory
The data directory contains the following files:

```
data
├── README.md
├── ls14.pkl
├── ls21.pkl
└── genesis.pkl
```

## Downloads
You can download the datasets from the Google Drive [PromptSub Datasets](https://drive.google.com/drive/folders/1r37uvvrKjbWNxbZvvO8kDpd1dm-S2qG1?usp=sharing).

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex

```
@inproceedings{lacerra-etal-2021-genesis,
    title = "{G}ene{S}is: {A} {G}enerative {A}pproach to {S}ubstitutes in {C}ontext",
    author = "Lacerra, Caterina  and
      Tripodi, Rocco  and
      Navigli, Roberto",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.844",
    doi = "10.18653/v1/2021.emnlp-main.844",
    pages = "10810--10823",
    abstract = "The lexical substitution task aims at generating a list of suitable replacements for a target word in context, ideally keeping the meaning of the modified text unchanged. While its usage has increased in recent years, the paucity of annotated data prevents the finetuning of neural models on the task, hindering the full fruition of recently introduced powerful architectures such as language models. Furthermore, lexical substitution is usually evaluated in a framework that is strictly bound to a limited vocabulary, making it impossible to credit appropriate, but out-of-vocabulary, substitutes. To assess these issues, we proposed GeneSis (Generating Substitutes in contexts), the first generative approach to lexical substitution. Thanks to a seq2seq model, we generate substitutes for a word according to the context it appears in, attaining state-of-the-art results on different benchmarks. Moreover, our approach allows silver data to be produced for further improving the performances of lexical substitution systems. Along with an extensive analysis of GeneSis results, we also present a human evaluation of the generated substitutes in order to assess their quality. We release the fine-tuned models, the generated datasets, and the code to reproduce the experiments at \url{https://github.com/SapienzaNLP/genesis}.",
}

@inproceedings{biemann-2012-turk,
    title = "Turk Bootstrap Word Sense Inventory 2.0: A Large-Scale Resource for Lexical Substitution",
    author = "Biemann, Chris",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/252_Paper.pdf",
    pages = "4038--4042",
    abstract = "This paper presents the Turk Bootstrap Word Sense Inventory (TWSI) 2.0. This lexical resource, created by a crowdsourcing process using Amazon Mechanical Turk (http://www.mturk.com), encompasses a sense inventory for lexical substitution for 1,012 highly frequent English common nouns. Along with each sense, a large number of sense-annotated occurrences in context are given, as well as a weighted list of substitutions. Sense distinctions are not motivated by lexicographic considerations, but driven by substitutability: two usages belong to the same sense if their substitutions overlap considerably. After laying out the need for such a resource, the data is characterized in terms of organization and quantity. Then, we briefly describe how this data was used to create a system for lexical substitutions. Training a supervised lexical substitution system on a smaller version of the resource resulted in well over 90{\%} acceptability for lexical substitutions provided by the system. Thus, this resource can be used to set up reliable, enabling technologies for semantic natural language processing (NLP), some of which we discuss briefly.",
}

@inproceedings{kremer-etal-2014-substitutes,
    title = "What Substitutes Tell Us - Analysis of an {``}All-Words{''} Lexical Substitution Corpus",
    author = "Kremer, Gerhard  and
      Erk, Katrin  and
      Pad{\'o}, Sebastian  and
      Thater, Stefan",
    booktitle = "Proceedings of the 14th Conference of the {E}uropean Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2014",
    address = "Gothenburg, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/E14-1057",
    doi = "10.3115/v1/E14-1057",
    pages = "540--549",
}

@inproceedings{lee-etal-2021-swords,
    title = "Swords: A Benchmark for Lexical Substitution with Improved Data Coverage and Quality",
    author = "Lee, Mina  and
      Donahue, Chris  and
      Jia, Robin  and
      Iyabor, Alexander  and
      Liang, Percy",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.345",
    doi = "10.18653/v1/2021.naacl-main.345",
    pages = "4362--4379",
    abstract = "We release a new benchmark for lexical substitution, the task of finding appropriate substitutes for a target word in a context. For writing, lexical substitution systems can assist humans by suggesting words that humans cannot easily think of. However, existing benchmarks depend on human recall as the only source of data, and therefore lack coverage of the substitutes that would be most helpful to humans. Furthermore, annotators often provide substitutes of low quality, which are not actually appropriate in the given context. We collect higher-coverage and higher-quality data by framing lexical substitution as a classification problem, guided by the intuition that it is easier for humans to judge the appropriateness of candidate substitutes than conjure them from memory. To this end, we use a context-free thesaurus to produce candidates and rely on human judgement to determine contextual appropriateness. Compared to the previous largest benchmark, our Swords benchmark has 3x as many substitutes per target word for the same level of quality, and its substitutes are 1.4x more appropriate (based on human judgement) for the same number of substitutes.",
}
```