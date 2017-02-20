Neural Verb Lexicon Model
=========================

Scripts for extracting context of source verbs (either window-based or following dependency relations).
Context is used as input to a feed-forward neural network (NPLM).

Different contexts for a Neural Verb Lexicon Model were evaluated for reranking nbest list produced by syntax-based MT system. Paper presented at IWSLT:

Maria Nadejde and Alexandra Birch and Philipp Koehn (2016): A Neural Verb Lexicon Model with Source-side Syntactic Context for String-to-Tree Machine Translation. Proceedings of the International Workshop on Spoken Language Translation (IWSLT). Seattle, Washington, USA

http://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_10.pdf


extract_verb_context.py:

Different factors can be defined via options (word, lemma, pos, dependency relation, particles).

The structure of the dependency context is defined via the argument: --dependency_context_config.
Format should be an integer for each of the following categories conected with _ (e.g 1_3_0_2_0_1): has_parent, no_children, no_grandchildren, no_prep_grandchildren, no_siblings, has_separate_pp

The source file (--parse_file) should be parsed with ParZU in CONLL format.
