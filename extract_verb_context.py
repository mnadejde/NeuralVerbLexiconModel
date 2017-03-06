# -*- coding: utf-8 -*-

# Script for extracting window and dependency context around the source and target verb
# Based on extract.py from bilingual-lm
# Creates a source and target vocabulary pruned by the vocabulary size
# Converts extract source and target n-grams to numberize format
# For source context m and target context n it creates n-grams of size: 2m+1 + n+1
# 2m source context words + source verb + n target context words + target verb
# special token added for particles


# example why dependency context might be good: "phenomen" should be in the context of gewonnen
# -> or just do prepreordering or create source context based on target context through alignments
# [u'dieses', u'ph\xe4nomen', u'hat', u'nach', u'den', u'wahlen', u'vom', u'november', u'2010', u'an', u'bedeutung', u'gewonnen' , u',', u'bei', u'denen', u'675', u'neue', u'republikanische', u'vertreter', u'in', u'26', u'staaten', u'verzeichnet', u'werden', u'konnten', u'.']
# context:
# [u'dieses', u'ph\xe4nomen', u'hat', u'nach', u'den', u'has']
# [u'an', u'bedeutung', u'gewonnen', u',', u'bei', u'gained']



from Util import print_dict_file2s
from Util import print_combine_file2s
import dependency as dp
import argparse
from collections import namedtuple
from collections import defaultdict
import codecs
import nltk
if nltk.__version__ == "2.0b9":
  from nltk.corpus import wordnet as wn
else:
  from nltk import wordnet as wn
from operator import itemgetter
from collections import Counter
import logging
import os.path

from Extract_verb_alignments import SentencePair
from Extract_verb_alignments import SelectSourceVerbs

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PART = "<part>"
NULL = "<null>"
NULL_CHILD = "<nullC>"
NULL_GRANDCHILD = "<nullGC>"
NULL_PPGRANDCHILD = "<nullPPGC>"
NULL_PARENT = "<nullP>"
NULL_SIBLING = "<nullS>"
NULL_PARTICLES = "<nullAVZ>"


class FACTOR_UNK:
  word = "<unk>"
  lemma = "<Lunk>"
  pos = "<Punk>"
  arc_label = "<Dunk>"
  subcat = "<Sunk>"

global total_unk

LOG = logging.getLogger(__name__)

### create, prune and save vocabularies ###

def load_vocabulary(f, size,delta):
  vocab_to_id = {}
  for i, line in enumerate(f):
    if i == size:
      return vocab_to_id
    word, id = line.decode("utf-8").strip().split()
    vocab_to_id[word] = int(id)
  return vocab_to_id

def get_vocabulary(f):
  vocab = Counter()
  for line in f:
    vocab.update(line.decode("utf-8").strip().split())
    vocab[BOS] +=1
    vocab[EOS] +=1
  return vocab

def get_vocabulary_from_parsed_data(f,factors, subcat, subcat_ordered):
  vocab = Counter()
  vocab_subcat = Counter()
  sentence_count = 0
  while True:
    # read dependency tree in CONLL format for each sentence
    dep_graph = dp.CoNLLXFormat.read_depgraph(f)
    if dep_graph == None:
      break
    sorted_tokens = dep_graph.sorted_tokens()
    for factor_name, factor in factors:
      vocab.update(map(factor, sorted_tokens))
      vocab[factor_name + BOS] +=1
      vocab[factor_name + EOS] +=1
    if subcat or subcat_ordered:
      for token in sorted_tokens:
        predicate = SelectSourceVerbs.is_predicate(token, False)
        if predicate:
          if subcat_ordered:
            vocab_subcat[get_subcat(token, lambda x: x.arc_label, "S", True)] += 1
          if subcat:
            vocab_subcat[get_subcat(token, lambda x: x.arc_label, "S", False)] += 1
    #vocab[BOS] +=1
    #vocab[EOS] +=1
    sentence_count += 1
  return vocab, vocab_subcat

def get_pruned_vocabulary(vocab, size):
  return Counter(dict(vocab.most_common(size)))

def get_vocabs_ids(svocab, subcat_vocab, tvocab, factors):
  #delete keywords from vocabularies
  del svocab[NULL]
  del svocab[NULL_CHILD]
  del svocab[NULL_GRANDCHILD]
  del svocab[NULL_PPGRANDCHILD]
  del svocab[NULL_PARENT]
  del svocab[NULL_SIBLING]
  del svocab[NULL_PARTICLES]
  del svocab[UNK]
  for factor_name, factor in factors:
    del svocab[factor_name + NULL_CHILD]
    del svocab[factor_name + NULL_GRANDCHILD]
    del svocab[factor_name + NULL_PPGRANDCHILD]
    del svocab[factor_name + NULL_PARENT]
    del svocab[factor_name + NULL_SIBLING]
  
  del tvocab[NULL]
  del tvocab[UNK]
  
  #create list of words in order of their frequencies
  svocab_list = [item[0] for item in svocab.most_common()]
  #subcat_vocab_list = [item[0] for item in subcat_vocab.most_common()]
  svocab_list += [item[0] for item in subcat_vocab.most_common()]
  tvocab_list = [item[0] for item in tvocab.most_common()]
  
  
  #add keywords to list of words
  svocab_list.insert(0, NULL)
  for factor_name, factor in factors:
    svocab_list.insert(0, factor_name + NULL_CHILD)
    svocab_list.insert(0, factor_name + NULL_GRANDCHILD)
    svocab_list.insert(0, factor_name + NULL_PPGRANDCHILD)
    svocab_list.insert(0, factor_name + NULL_PARENT)
    svocab_list.insert(0, factor_name + NULL_SIBLING)
  svocab_list.insert(0,NULL_PARTICLES)
  svocab_list.insert(0, UNK)


  tvocab_list.insert(0, NULL)
  tvocab_list.insert(0, UNK)
  
  #create vocab dictionary mapping to ids
  svocab_to_id = {}
  tvocab_to_id = {}
  # target words ids -> nplm wants smaller ids for target (output vocab) so it correctly determines the size of vocabulary
  for i,word in enumerate(tvocab_list):
    tvocab_to_id[word] = i
  # source words ids start after the last target id
  delta = len(tvocab_to_id) #tvocab_list)
  for i,word in enumerate(svocab_list):
    svocab_to_id[word] = i + delta

  #delta += len(svocab_to_id) #svocab_list)
  #for i,word in enumerate(subcat_vocab_list):
    #svocab_to_id[word] = i + delta

  return svocab_to_id, tvocab_to_id


# for source context ml=m, mr=m and for target context ml=n, mr=0
# => cotnext size: ml+mr+1
def get_window_context(tokens, ml,mr, spos, sorted_tokens, factor, factor_name):
  context = []
  factored_context = []
  # padding to at the begining of sentence
  for i in range(max(0, ml - spos)):
    context.append(factor_name + BOS)
  # words in window of size m
  if sorted_tokens and factor:
    base = False
    if base:
      factored_context += [s for s in sorted_tokens[max(0, spos - ml):spos + mr + 1]]
    # add only the words surrounding not the token itself
    else:
      factored_context += [s for s in sorted_tokens[max(0, spos - ml):spos]]
      factored_context += [s for s in sorted_tokens[spos+1:spos + mr + 1]]
    
    #if factored_context:
    #  print [s for s in tokens[max(0, spos - ml):spos + mr + 1]]
    #  for j,w in enumerate(factored_context):
    #    print j, w.word
    context += map(factor, factored_context)
  else:
    # !!!!! PROBLEM !!!! - there is mismatch between vocabulary of the source tokens and that of the parsed source sentences
    # !!!! - deescaped special charcters &apos; &quot; etc
    context += [s for s in tokens[max(0, spos - ml):spos + mr + 1]]
  # padding to the end of the sentence
  for i in range(max(0, spos + mr + 1 - len(tokens))):
    context.append(factor_name + EOS)
  return context

# this is wrong -> should change vocab from dict to list and return the id in the list
def word_to_id(word, vocab):
  global total_unk
  if vocab.has_key(word):
    return vocab[word]
  else:
    total_unk += 1
    return vocab[UNK]


def get_numberized_context(context, svocab, tvocab, src_win, trg_win):
  numberized_context = []
  for item in context:
    ngram = []
    if len(item) != src_win+trg_win:
      LOG.error("context size doesn't match: %s %d %d" % item, src_win, trg_win)
      exit(-1)
    for word in item[:src_win]:
      ngram.append(word_to_id(word, svocab))
    for word in item[-(trg_win):]:
      ngram.append(word_to_id(word, tvocab))
    numberized_context.append(ngram)
  return numberized_context



def adjust_context(context, dep_win, padding):
  if len(context) < dep_win:
    for i in range(len(context),dep_win):
      context.append(padding)
    return context
  else:
    return context[:dep_win]

def get_factor(context, factor):
  return map(factor, context)

# returns a concatenation of the children
# in principal to be used with arc_label factor in order to obtain the subcategorization frame
def get_subcat(token, factor, factor_name, unordered):
  context = []
  children = []
  except_id = -1
  if token.get_dependents_by_label("aux"):
    return "S_aux"
  if token.arc_label == "aux" and token.head and token.head.position !=0:
    except_id = token.position # problem with the main verb being added as a child
    token = token.head
  for child, r in token.get_dependents():
    if child.position != except_id:
      children.append(child)
  if unordered:
    context = map(factor, sorted(children, key= lambda x: x.arc_label))
  else:
    context = map(factor, sorted(children, key= lambda x: x.position))
  return "_".join([factor_name] + context)


# returns children of the given head token
# children are sorted according to the position in the sentence.
# we do not distinguish between left and right children as german has flexible word order (and the main predicate is likely to be at the end of the sentence)
# we return only dep_win number of children. if less children are found we pad with NULL tokens
def get_children(token, dep_win, except_id, except_pp, except_avz, separate_children, factor, factor_name):
  context = []
  children = []
  main_children = []
  other_children = []
  for child, r in token.get_dependents():
    # add the preposition later - with the prepositional modifier as a grandchild
    if except_pp and (child.arc_label == "pp" or child.arc_label=="objp"):
      continue
    # add particle as separate context
    if except_avz and (child.arc_label == "avz"):
      continue
    if child.position != except_id:
      if separate_children:
        if child.arc_label in SelectSourceVerbs.child_labels:
          main_children.append(child) # "obja","objc","objd", "objg", "obji","subj", "subjc"
        else:
          other_children.append(child)
      else:
        children.append(child)
  #context = map(lambda x: x[0].word,sorted(token.get_dependents(), key= lambda x: x[0].position))
  #context = map(lambda x: x.word,sorted(children, key= lambda x: x.position))
  #context = get_factor(sorted(children, key= lambda x: x.position),lambda x: x.word)
  if separate_children:
    children += sorted(main_children, key= lambda x: x.position)
    children += sorted(other_children, key= lambda x: x.position)
    context = map(factor, children)
  else:
    context = map(factor, sorted(children, key= lambda x: x.position))
  return adjust_context(context, dep_win, factor_name + NULL_CHILD)

def get_grandchildren(token, dep_win, factor, factor_name):
  context = []
  for child, r in token.get_dependents():
    #context += map(lambda x: x[0].word,sorted(child.get_dependents(), key= lambda x: x[0].position))
    context += map(factor, map(lambda y: y[0], sorted(child.get_dependents(), key= lambda x: x[0].position)))
  return adjust_context(context, dep_win, factor_name + NULL_GRANDCHILD)

# problem -> OBJP - PREPOSITIONAL OBJECT -> prepositional phrase that is complement -> Er ging mit ihr *ins* Kino
def get_prep_grandchildren(token, dep_win, with_pp, factor, factor_name):
  context = []
  for child, r in token.get_dependents():
    if child.arc_label == "pp" or child.arc_label == "objp": # preposition is the head in the ParZu grammar
      if with_pp:
        #context.append(child.word)
        context.append(child)
      #context += map(lambda x: x[0].word, child.get_dependents())
      context += map(lambda y: y[0], child.get_dependents())
  context = map(factor, context)
  return adjust_context(context, dep_win, factor_name + NULL_PPGRANDCHILD)

# return the head word or NULL
# since we're dealing with verbs it doesn't make sense to look for grandparents
def get_parent(token, factor, factor_name):
  if token.head and token.head.position !=0 :
    return map(factor, [token.head])
  else:
    return [factor_name + NULL_PARENT]

def get_siblings(token, dep_win, factor, factor_name):
  context = []
  if token.head and token.head.position !=0 :
    for child,r in token.head.get_dependents():
      if child.position != token.position:
        context.append(child)
  context = map(factor, context)
  context = adjust_context(context, dep_win, factor_name + NULL_SIBLING)

class DepContext:
  def __init__(self, win_parent, win_children, win_grandchildren, win_prep_grandchildren, win_siblings, separate_pp, subcat, subcat_ordered, particles, separate_children):
    self.win_parent = win_parent
    self.win_children = win_children
    self.win_grandchildren = win_grandchildren
    self.win_prep_grandchildren = win_prep_grandchildren
    self.win_siblings = win_siblings
    self.separate_pp = separate_pp
    self.subcat = subcat
    self.subcat_ordered = subcat_ordered
    self.particles = particles
    self.separate_children = separate_children
    if self.separate_pp:
      self.win_prep_grandchildren *= 2 # twice as many words since we add the preposition to the prepositional modifier
    self.win_total = self.win_parent + self.win_children + self.win_grandchildren + self.win_prep_grandchildren + self.win_siblings #  + 1 -> for souce token

# PROBLEM: the auxiliary verb is considered the head and the main verb the child. all the arguments get attached to the auxiliary verb
def get_verb_dep_context(token, win_parent, win_children, win_grandchildren, win_prep_grandchildren, win_siblings, separate_pp, separate_avz, separate_children, factor, factor_name):
  context = []
  #context += map(factor, [token])
  if win_parent:
    context += get_parent(token, factor, factor_name)
  
  # if it's main verb with arc_label aux then all the arguments are attached to the actual auxiliary
  except_id = -1
  if token.arc_label == "aux" and token.head and token.head.position !=0:
    except_id = token.position # problem with the main verb being added as a child
    token = token.head

  if win_children:
    context += get_children(token, win_children, except_id, separate_pp, separate_avz, separate_children, factor, factor_name)
  if win_grandchildren:
    context += get_grandchildren(token, win_grandchildren, factor, factor_name)
  if win_prep_grandchildren:
    context += get_prep_grandchildren(token, win_prep_grandchildren, separate_pp, factor, factor_name)
  if win_siblings:
    context += get_siblings(token, win_siblings, factor, factor_name)
  return context



# traverse children and grandchildren up to dep_count
# save context in dep_win
def get_verb_dependents_context1(token, dep_win): #dep_win, dep_count):
  children = map(itemgetter(0),sorted(token.get_dependents(), key= lambda x: x[0].position))
  # add children to context
  context = map(lambda x: x.word, children)
  #print token.word
  #for child in children:
    #print child.word, child.arc_label
  # add parent to context
  if token.head and token.head.position !=0 :
    head = token.head
    context.append(head.word)
    # add siblings to context
    for child,r in head.get_dependents():
      if child.position != token.position:
        context.append(child.word)
  if len(context) >= dep_win:
    return context[:dep_win]+[token.word]
  
  if len(context) < dep_win:
    grandchildren = []
    # add grandchildren
    for child in children:
      grandchildren += map(lambda x: x[0].word,sorted(child.get_dependents(), key= lambda x: x[0].position))
    context = context + grandchildren
    if len(context) >= dep_win:
      return context[:dep_win]+[token.word]
  
  for i in range(len(context),dep_win):
    context.append(NULL)
  return context[:]+[token.word]


def get_verb_context(trg_unk, trg_vb_only, main_only, dep_graph, sentence_pair, src_win, trg_win, dep_context, subcat, subcat_ordered, src_particles, context_type, factors):
  ngrams = []
  ids = []
  sorted_tokens = dep_graph.sorted_tokens() #[1:]
  for i, token  in enumerate(dep_graph.tokens):
    predicate = False
    particles = []
    # if token is a verb
    predicate = SelectSourceVerbs.is_predicate(token, main_only)
    if predicate:
      context = []
      particles = SelectSourceVerbs.get_particles(token)
      trg_words = sentence_pair.get_aligned_trg_words(i-1)
      trg_tags = sentence_pair.get_aligned_trg_tags(i-1)
      # source context
      for factor_name, factor in factors:
        base = False
        if not base:
          context += map(factor, [token])
        if (context_type == 1 or context_type == 2) and dep_context:
          #factor = lambda x: x.arc_label
          context += get_verb_dep_context(token, dep_context.win_parent, dep_context.win_children, dep_context.win_grandchildren, dep_context.win_prep_grandchildren, dep_context.win_siblings, dep_context.separate_pp, dep_context.particles, dep_context.separate_children, factor, factor_name)
          #context = get_verb_dependents_context(token, src_win * 2)
          #print context
        if (context_type == 0 or context_type == 2):
          context += get_window_context(sentence_pair.src_tokens, src_win, src_win, i-1, sorted_tokens, factor, factor_name) # + trg_words
      if subcat:
        context.append(get_subcat(token, lambda x: x.arc_label, "S", False))
      # order the labels lexicografically -> less data sparsity due to free word order in german; highlights the subcatecorization frame but not the order of arguments
      if subcat_ordered:
        context.append(get_subcat(token, lambda x: x.arc_label, "S", True))
      if src_particles:
        if particles:
          context.append(particles[0])
        else:
          context.append(NULL_PARTICLES)
      
      # target context -> should select midpoint or verb
      #trg_id = sentence_pair.alignments[i-1][0]
      trg_id = SelectSourceVerbs.find_aligned_verb_id(sentence_pair, i, trg_vb_only)
      if trg_id is None and not trg_unk:
        continue
			# this is only for when you're trying to predict a target verb given a souce context; not for training or evaluating
      if trg_id is None and trg_unk:
        context += [UNK]
      else:
        context += get_window_context(sentence_pair.trg_tokens, trg_win, 0, trg_id, None, None, None)
      #print context
      #print ""
      ngrams.append(context)
      ids.append(token.position)
  return ngrams, ids

def save_context(context,f,numberized):
  if context and f:
    for ngram in context:
      if numberized:
        line = ' '.join([str(id) for id in ngram])
      else:
        line = ' '.join([word for word in ngram])
      print>>f, line
      #for id in ngram:
        #print>>f, id, " ",
      #print>>f, ""

def save_vocab(vocab, f):
  if vocab and f:
    #for word in vocab.keys():
      #print>>f, word, vocab[word]
    count = 0
    items = sorted(vocab.items(), key= lambda x: x[1])
    for word,i in items:
      count += 1
      print>>f, word.encode("utf-8"), i
    LOG.info("Wrote dictionary to file: %d" % count)

def main():
  logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

  global stats
  parser = argparse.ArgumentParser()
  parser.add_argument('--parse_file',type=str)
  parser.add_argument('--alignment_file', type=str)
  parser.add_argument('--source_file', type=str)
  parser.add_argument('--target_file', type=str)
  parser.add_argument('--target_pos_file', type=str)
  parser.add_argument('--out_ngrams', type=str)
  parser.add_argument('--working_dir', type=str)
  parser.add_argument('--vocab_name', type=str)
  parser.add_argument('--model_name', type=str)
  parser.add_argument('-n', '--target-context', type=int, dest='n')
  parser.add_argument('-m', '--source-context', type=int, dest='m')
  parser.add_argument('-s', '--source-vocab-size', type=int, dest='sprune')
  parser.add_argument('-t', '--target-vocab-size', type=int, dest='tprune')
  parser.add_argument('--subcat-vocab-size', type=int, dest='scprune')
  parser.add_argument('--load_vocab',action='store_true')
  parser.add_argument('--output_sentence_ids',action='store_true')
  #parser.add_argument('--dependency_context',action='store_true')
  parser.add_argument('-c','--context_type', type=int)
  parser.add_argument('--dependency_context_config', type=str)
  parser.add_argument('--factor_word',action='store_true')
  parser.add_argument('--factor_lemma',action='store_true')
  parser.add_argument('--factor_dep_rel',action='store_true')
  parser.add_argument('--factor_pos',action='store_true')
  parser.add_argument('--subcat',action='store_true')
  parser.add_argument('--subcat_ordered',action='store_true')
  parser.add_argument('--particles',action='store_true')
  parser.add_argument('--separate_children',action='store_true')
  parser.add_argument('--nbest',action='store_true')
  parser.add_argument('--main_only',action='store_true')
  parser.add_argument('--trg_vb_only',action='store_true')
  parser.add_argument('--trg_unk',action='store_true') # when I'm predicting using a trained model based on source context I don't care what the target verb is
  
  parser.set_defaults(
    parse_file='/PredictVerbs/newstest201345.input.split.9.parse', #test.nbest.parse'
    alignment_file='/PredictVerbs/newstest201345.output.9.trace.align.s-t',
    source_file='/PredictVerbs/newstest201345.output.9.trace.align.de', #test.nbest.de',
    target_file='/PredictVerbs/newstest201345.output.9.trace.align.en', #test.nbest.en', #
    target_pos_file='/PredictVerbs/newstest201345.output.9.trace.align.en.pos', #test.nbest.pos', #
    out_ngrams='ngrams1',
    n=0,
    m=2,
    working_dir='/PredictVerbs/',
    vocab_name='vocab',
    sprune=5000,
    tprune=5000,
    scprune = 50000,
    load_vocab=False,
    context_type=0,
    dependency_context_config='0_3_0_1_0_1',
    factor_word=True,
    factor_lemma=False,
    factor_dep_rel=False,
    factor_pos=False,
    subcat=False,
    subcat_ordered=False,
    particles=False,
    separate_children=False,
    output_sentence_ids=False,
    nbest=False,
    main_only=False,
    trg_vb_only=False,
    trg_unk=False
    )
  
  args = parser.parse_args()
  LOG.info("Arguments: %s" % args)

  fp = open(args.parse_file,'r')
  fa = None
  if not args.nbest and args.alignment_file:
    fa = open(args.alignment_file,'r')
  fs = open(args.source_file,'r')
  ft = open(args.target_file,'r')
  ftp = open(args.target_pos_file,'r')

  #output file for numberized context ngrams
  #fo = open(args.working_dir+args.out_ngrams, 'w')
  #if os.path.exists(args.working_dir+args.out_ngrams+'.numberized') or os.path.exists(args.working_dir+args.out_ngrams):
  #    LOG.error('You cannot write to files that already exist - check:')
  #    LOG.error('%s' % args.working_dir+args.out_ngrams)
  #    LOG.error('%s' % args.working_dir+args.out_ngrams+'.numberized')
  #    exit(-1)
  
  fon = open(args.working_dir+args.out_ngrams+'.numberized', 'w')
  fsid = None
  if (args.output_sentence_ids):
    fsid = open(args.working_dir+args.out_ngrams+'.sid', 'w')
  fvs = None #open(args.working_dir+args.vocab_name+'.src','w')
  fvt = None #open(args.working_dir+args.vocab_name+'.trg', 'w')
  fo = codecs.open(args.working_dir+args.out_ngrams, encoding='utf-8', mode = 'w', errors='ignore')
  #fvs = codecs.open(args.working_dir+args.vocab_name+'.src', encoding='utf-8', mode = 'w', errors='ignore')
  #fvt = codecs.open(args.working_dir+args.vocab_name+'.trg', encoding='utf-8', mode = 'w', errors='ignore')
  
  dep_context = None
  # 0 - window, 1 - dependency, 2 - both
  if (args.context_type == 1 or args.context_type == 2):
    win_parent, win_children, win_grandchildren, win_prep_grandchildren, win_siblings, separate_pp = map(int,args.dependency_context_config.split('_'))
    dep_context = DepContext(win_parent, win_children, win_grandchildren, win_prep_grandchildren, win_siblings, separate_pp, args.subcat, args.subcat_ordered, args.particles, args.separate_children)
  
  # factors
  #factors = [lambda x: x.word, lambda x: x.arc_label]
  factors = []
  if args.factor_word:
    #factors.append(("W", lambda x: "W_"+x.word))
    factors.append(("W", lambda x: x.word))
  if args.factor_lemma:
    #factors.append(("L", lambda x: NULL if x.lemma is None else "L_"+x.lemma))
    factors.append(("L", lambda x: NULL if x.lemma is None else x.lemma))
  if args.factor_dep_rel:
    factors.append(("D", lambda x: "D_"+x.arc_label))
  if args.factor_pos:
    factors.append(("P",lambda x: NULL if x.pos is None else "P_"+x.pos))

  if not factors:
    LOG.error("No factors. Need at least one factor for extracting word context")
    exit(-1)
  
  #create, prune and save vocabularies
  if (args.load_vocab):
    LOG.info("Loading vocabularies")
    fvs = open(args.working_dir+args.vocab_name+'.src','r')
    fvt = open(args.working_dir+args.vocab_name+'.trg', 'r')
    #fvs.seek(0)
    #fvt.seek(0)
    tvocab_to_id = load_vocabulary(fvt, args.tprune, 0)
    svocab_to_id = load_vocabulary(fvs, args.sprune, len(tvocab_to_id.keys()))
    LOG.info("Source types found: %d" % len(svocab_to_id.keys()))
    LOG.info("Target types found: %d" % len(tvocab_to_id.keys()))
  else:
    if args.nbest:
      LOG.error("nbest option only works with a provided vocabulary")
      exit(-1)
    LOG.info("Creating vocabularies")
    fvs = open(args.working_dir+args.vocab_name+'.src','w')
    fvt = open(args.working_dir+args.vocab_name+'.trg', 'w')
    
    #src_vocab = get_vocabulary(fs)
    src_vocab, subcat_vocab = get_vocabulary_from_parsed_data(fp, factors, args.subcat, args.subcat_ordered)
    trg_vocab = get_vocabulary(ft)
    LOG.info("Source types found: %d" % len(src_vocab))
    LOG.info("Subcat types found: %d" % len(subcat_vocab))
    LOG.info("Target types found: %d" % len(trg_vocab))
    
    src_vocab = get_pruned_vocabulary(src_vocab, args.sprune)
    subcat_vocab = get_pruned_vocabulary(subcat_vocab, args.scprune)
    trg_vocab = get_pruned_vocabulary(trg_vocab, args.tprune)
    
    svocab_to_id, tvocab_to_id = get_vocabs_ids(src_vocab, subcat_vocab, trg_vocab, factors)
    
    LOG.info("Source types remaining: %d" % len(svocab_to_id.keys()))
    LOG.info("Target types remaining: %d" % len(tvocab_to_id.keys()))

    save_vocab(svocab_to_id, fvs)
    save_vocab(tvocab_to_id, fvt)

  if fvs and fvt:
    fvs.close()
    fvt.close()
  
  #rewind file handle
  fs.seek(0)
  ft.seek(0)
  fp.seek(0)
  
  LOG.info("Extracting verb contexts")
  sentence_count = 0
  global total_unk
  total_unk = 0
  # read files line by line and process
  while True:
    # read dependency tree in CONLL format for each sentence
    dep_graph = dp.CoNLLXFormat.read_depgraph(fp)
    if dep_graph == None:
      break

    sentence_pairs = []
    if args.nbest:
      sentence_pairs = SentencePair.read_nbest_separate(fs,ft,ftp,None, sentence_count)
      # get_verb_context_from_nbest(dep_graph, sentence_pairs, args.m, args.n, dep_context, args.context_type, factors, args.subcat)
    else:
      sentence_pair = SentencePair()
      sentence_pair.read_sentence_pair_separate(fs, ft, fa, ftp, False)
      sentence_pairs.append(sentence_pair)

    sentence_count += 1

    for ni, sentence_pair in enumerate(sentence_pairs):
      #print sentence_pair.src_tokens
      #print sentence_pair.trg_tokens
      #print sentence_pair.trg_tags
      #print sentence_pair.alignments
      #sentence_pair.print_all_aligned_words()
      context, src_ids = get_verb_context(args.trg_unk, args.trg_vb_only, args.main_only, dep_graph, sentence_pair, args.m, args.n, dep_context, args.subcat, args.subcat_ordered, args.particles, args.context_type, factors)
      # window context
      if args.context_type == 0:
        numberized_context = get_numberized_context(context, svocab_to_id, tvocab_to_id, (2*args.m + 1) * len(factors) + args.subcat + args.subcat_ordered + args.particles, args.n + 1)
      # dependency path context
      if args.context_type == 1:
        numberized_context = get_numberized_context(context, svocab_to_id, tvocab_to_id, (dep_context.win_total + 1) * len(factors) + args.subcat + args.subcat_ordered + args.particles, args.n + 1)
      # both window and dependency path context
      if args.context_type == 2:
        numberized_context = get_numberized_context(context, svocab_to_id, tvocab_to_id, dep_context.win_total * len(factors) + (2*args.m + 1) * len(factors) + args.subcat + args.subcat_ordered + args.particles, args.n + 1)
      
      #if dep_context:
      #  numberized_context = get_numberized_context(context, svocab_to_id, tvocab_to_id, dep_context.win_total * len(factors) + args.subcat, args.n + 1)
      #else:
      #  numberized_context = get_numberized_context(context, svocab_to_id, tvocab_to_id, (2*args.m + 1) * len(factors) + args.subcat, args.n + 1)
      
      save_context(context, fo, False)
      save_context(numberized_context, fon, True)
      #print context
      #print numberized_context

      if fsid:
        for k in range(len(src_ids)):
          print>>fsid, str(sentence_count), str(ni+1), src_ids[k]
      #if ni == 3:
        #exit(0)
    #if sentence_count == 3:
      #exit(0)
    
  LOG.info("Total unknown words: %d" %total_unk)
  LOG.info("Finished")

if __name__ == '__main__':
  
  main()
