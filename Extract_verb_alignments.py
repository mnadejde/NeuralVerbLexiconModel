# takes as input
# 1. a file produced by Read_trace_alignments.py containing source sentence, target sentence and word alignment (one per line)
# 2. and a similar file resulted from word aligning the source and reference translations
# 3. dependency parses of the source sentences (containing information about pos and label)
# For verbs not labeled as (aux, dep etc. - different for German) read target/reference alignments and see if words match. Compute precision.
# Problem: how to identify source verbs that are not auxiliaries. The Parzu syntactic annotation labels the main verb as aux dependent to the auxiliary verb
# From parzu documentation (https://github.com/rsennrich/ParZu/blob/master/LABELS.md):
# auxiliary verb relation (note that the auxiliary verb is the head of the relation)
# selecting verbs:
#   if pos starts with V and label is aux -> main verb attaching to auxiliary
#   if pos starts with V and has a child labeled: OBJ(A/C/D/G/I/P)
# selecting particles - meaning of the verb defined by moving particle:
#   has child with label "avz"
#   has child with label "zu" and child is "zu"
# selecting predicative nouns:
#   if pos starts with N and label is pred

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
import gzip
import re
import sys
import scipy.stats as ss
import numpy

def map_pos(pos):
  if pos.startswith('N'):
    return 'n'
  if pos.startswith('V'):
    return 'v'
  if pos.startswith('J'):
    return 'a'
  if pos in ['RB', 'RBR', 'RBS']:
    return 'r'
  return None

lm = nltk.stem.WordNetLemmatizer()

class SentencePair:
  def __init__(self):
    self.src_tokens = []
    self.trg_tokens = []
    self.alignments = defaultdict(list)
    self.trg_tags = []
  
  def init(self, src_tokens, trg_tokens, alignments, trg_tags):
    self.src_tokens = src_tokens
    self.trg_tokens = trg_tokens
    self.alignments = alignments
    self.trg_tags = trg_tags
    
  @staticmethod
  def read_alignment(line):
    # source to target alignment dictionary. allows one to many alignments
    alignments = defaultdict(list)
    for align in map(lambda x: x.split('-'),line.split()):
      alignments[int(align[0])].append(int(align[1]))
    return alignments

  @staticmethod
  def read_nbest_alignment(line, n, m):
    # source to target alignment dictionary. allows one to many alignments
    alignments = defaultdict(list)
    for align in map(lambda x: x.split('-'),line.split()):
      # skip <s> </s>
      if int(align[0]) == 0 or int(align[1])== 0 or int(align[0]) == m or int(align[1]) == n:
        continue
      alignments[int(align[0])-1].append(int(align[1])-1)
    return alignments

  @staticmethod
  def read_lines(f, st, pos_available, lemma_available):
    global lm
    src_tokens = f.readline().decode("utf-8").lower().split()
    trg_tokens = f.readline().decode("utf-8").lower().split()
    #for word,pos in nltk.pos_tag(s.split()):
    if pos_available:
      trg_pos = f.readline().decode("utf-8").split()
    elif st:
      trg_pos = [pos for word, pos in st.tag(trg_tokens)]
    else:
      trg_pos = [pos for word,pos in nltk.pos_tag(trg_tokens)]
    
    if lemma_available:
      trg_lemma = f.readline().decode("utf-8").split()
    else:
      trg_lemma = [lm.lemmatize(word,map_pos(pos)) if map_pos(pos) else lm.lemmatize(word) for word, pos in zip(trg_tokens, trg_pos)]
    
    if len(trg_pos) != len(trg_lemma):
      print "Error: tag pos and tag lemma lists don't match"
      exit(0)
    
    tags = zip(trg_pos, trg_lemma)
    
    #if st:
     # tags = [(pos,lm.lemmatize(word,map_pos(pos))) if map_pos(pos) else (pos,lm.lemmatize(word)) for word,pos in st.tag(trg_tokens)]
    #else:
     # tags = [(pos,lm.lemmatize(word,map_pos(pos))) if map_pos(pos) else (pos,lm.lemmatize(word)) for word,pos in nltk.pos_tag(trg_tokens)]
    
    alignments = SentencePair.read_alignment(f.readline())
    f.readline() # empty line
    return src_tokens, trg_tokens, alignments, tags

  @staticmethod
  def read_lines_separate(s,t,a,p, lowercase):
    global lm
    if lowercase:
      src_tokens = s.readline().decode("utf-8").lower().split()
      trg_tokens = t.readline().decode("utf-8").lower().split()
    else:
      src_tokens = s.readline().decode("utf-8").split()
      trg_tokens = t.readline().decode("utf-8").split()
    # POS file -> it seems using lemma and nltk.pos_tag is slow
    pos_tags = p.readline().split()
    tags = [(pos, None) for pos in pos_tags]
    #tags = [(pos,lm.lemmatize(word,map_pos(pos))) if map_pos(pos) else (pos,lm.lemmatize(word)) for word,pos in nltk.pos_tag(trg_tokens)]
    
    alignments = SentencePair.read_alignment(a.readline())
    return src_tokens, trg_tokens, alignments, tags
    
  @staticmethod
  def read_nbest_separate(s,n,p,l, sid):
    global lm
    src_tokens = s.readline().decode("utf-8").split()
    sentence_pairs = []
    while True:
      last_pos = n.tell()
      nbest_fields = n.readline().decode("utf-8").split("|||")
      if  nbest_fields[0] == '' or int(nbest_fields[0])!= sid:
        n.seek(last_pos)
        return sentence_pairs
      trg_tokens = nbest_fields[1].split()
      pos_tags = p.readline().split()
      if l:
        lemma_tags = l.readline().split()
        tags = zip(pos_tags, lemma_tags)
      else:
        tags = [(pos, None) for pos in pos_tags]
      #alignments include start and end symbol - ignore them and adjust ids
      alignments = SentencePair.read_nbest_alignment(nbest_fields[4], len(trg_tokens)+1, len(src_tokens)+1)
      new_sentence_pair = SentencePair()
      new_sentence_pair.init(src_tokens, trg_tokens, alignments, tags)
      sentence_pairs.append(new_sentence_pair)
  

  def read_sentence_pair(self, f, st, pos_available, lemma_available):
    self.src_tokens, self.trg_tokens, self.alignments, self.trg_tags = self.read_lines(f,st, pos_available, lemma_available)
  
  def read_sentence_pair_separate(self, s,t,a,p, lowercase):
    self.src_tokens, self.trg_tokens, self.alignments, self.trg_tags = self.read_lines_separate(s,t,a,p, lowercase)

  def get_aligned_trg_words(self, id):
    trg_words = map(lambda x: self.trg_tokens[x], sorted(self.alignments[id]))
    return trg_words
  
  def get_aligned_trg_tags(self, id):
    if not self.alignments.has_key(id):
      print id
      print self.alignments
      exit(-1)
    for t in self.alignments[id]:
      if t >= len(self.trg_tags):
        print t, len(self.trg_tags)
        print self.alignments
    trg_tags = map(lambda x: self.trg_tags[x], sorted(self.alignments[id]))
    return trg_tags

  def print_all_aligned_words(self):
    for i in range(len(self.src_tokens)):
      print self.src_tokens[i], self.get_aligned_trg_words(i)

class SelectSourceVerbs:
  # a predicate verb might have a child with one of these labels (parzu labels)
  child_labels = ["obja","objc","objd", "objg", "obji", "objp", "subj", "subjc"]
  # some verbs have particles that change the meaning of the verb and often are dropped/not translated
  particle_labels = ["avz"] #, "part"]
  # some main verbs are attached to the auxiliary verb and labeled as AUX
  verb_labels = ["aux"]
  # some nouns are predicative
  noun_labels = ["pred"]
  # dictionary of all seen source verbs - will be used to filter the rule table
  source_verbs_types = defaultdict(int)
  # dictionary of all aligned trace verb types
  trace_verbs_types = defaultdict(int)
  # dictionary of all aligned ref verb types
  ref_verbs_types = defaultdict(int)
  
  def __init__(self):
    self.dep_graph = None
    self.sentence_pair_trace = None
    self.sentence_pair_ref = None

  @staticmethod
  def update_verb_dictionaries(source_verbs):
    for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
      SelectSourceVerbs.source_verbs_types[vb.word]+=1
      for token, tags in zip(trace_tokens, trace_tags):
        if tags[0].startswith('V'):
          SelectSourceVerbs.trace_verbs_types[token] += 1
      for token, tags in zip(ref_tokens, ref_tags):
        if tags[0].startswith('V'):
          SelectSourceVerbs.ref_verbs_types[token] += 1

  @staticmethod
  def is_predicate(token, main_only):
    predicate = False
    # if token is a verb
    if not token.isdummy and token.pos[0] == 'V':
      # main verb attaced to auxiliary verb
      if token.arc_label in SelectSourceVerbs.verb_labels:
        predicate = True
      for child,r in token.get_dependents():
        if r in SelectSourceVerbs.child_labels:
          predicate = True
          #return (i,token)
          break
      #in the dep parse the auxiliary is the head and the main verb the child ??
      if(main_only):
        for child,r in token.get_dependents():
          if r == "aux":
            predicate = False
    return predicate
  
  # for a given source id find the corresponding verb in the aligned target words
  @staticmethod
  def find_aligned_verb_id(sentence_pair, src_id, vb_only):
    if not len(sentence_pair.alignments[src_id-1]):
      return None
    vaux_trg_ids = []
    # not be, has, do: VV, VVD, VVG,...
    vv_trg_ids = []
    for trg_id in sentence_pair.alignments[src_id-1]:
      # first tag is POS -> problem with auxiliaries
      if sentence_pair.trg_tags[trg_id][0].startswith('V'):
        # assumes TreeTagger pos tags: http://courses.washington.edu/hypertxt/csar-v02/penntable.html
        if sentence_pair.trg_tags[trg_id][0].startswith('VV'):
          vv_trg_ids.append(trg_id)
        else:
          vaux_trg_ids.append(trg_id)
        #return trg_id
    # return first verb tag that is not be*, has*, do*
    if len(vv_trg_ids) > 0:
      return vv_trg_ids[0]
    elif len(vaux_trg_ids) > 0:
      return vaux_trg_ids[0]
    # if no verb tag return mid position - should check for noun or not return anything -> problem with commas
    if not vb_only:
      midpoint = (len(sentence_pair.alignments[src_id-1])-1)/2
      return sentence_pair.alignments[src_id-1][midpoint]
    else:
      return None

  @staticmethod
  def get_particles(token):
    particles = []
    for label in SelectSourceVerbs.particle_labels:
      child = token.get_dependents_by_label(label)
      if child:
        particles.append(child[0].word)
    return particles

  @staticmethod
  def process_sentence_pair(main_only_src, main_only_trg, dep_graph, sentence_pair_trace, sentence_pair_ref):
    source_verbs = []
    #main_only = False
    for i, token  in enumerate(dep_graph.tokens):
      predicate = False
      particles = []
      # if token is a verb
      predicate = SelectSourceVerbs.is_predicate(token, main_only_src)
      if predicate:
        particles = SelectSourceVerbs.get_particles(token)
        
        # i=0 is root dummy node
        trg_words_trace = []
        trg_words_ref = []
        trg_tags_trace = []
        trg_tags_ref = []
        if main_only_trg:
          trg_vb_id_trace = SelectSourceVerbs.find_aligned_verb_id(sentence_pair_trace, i, True)
          trg_vb_id_ref = SelectSourceVerbs.find_aligned_verb_id(sentence_pair_ref, i, True)
          if trg_vb_id_trace:
            trg_words_trace = [sentence_pair_trace.trg_tokens[trg_vb_id_trace]]
            trg_tags_trace = [sentence_pair_trace.trg_tags[trg_vb_id_trace]]
          if trg_vb_id_ref:
            trg_words_ref = [sentence_pair_ref.trg_tokens[trg_vb_id_ref]]
            trg_tags_ref = [sentence_pair_ref.trg_tags[trg_vb_id_ref]]
        else:
          trg_words_trace = sentence_pair_trace.get_aligned_trg_words(i-1)
          trg_words_ref = sentence_pair_ref.get_aligned_trg_words(i-1)
          trg_tags_trace = sentence_pair_trace.get_aligned_trg_tags(i-1)
          trg_tags_ref = sentence_pair_ref.get_aligned_trg_tags(i-1)
        source_verbs.append((token,trg_words_trace,trg_words_ref,particles, trg_tags_trace, trg_tags_ref))
    # count verb types - source, trace, reference
    SelectSourceVerbs.update_verb_dictionaries(source_verbs)
    return source_verbs

def print_pair(source_verbs, sentence_pair_trace, sentence_pair_ref, out_file, s_id):
  src_trace = ' '.join(sentence_pair_trace.src_tokens)
  src_ref = ' '.join(sentence_pair_ref.src_tokens)
  # problem -> unaligned source words are not extracted from the trace (from rules that drop words)
  if src_trace != src_ref:
    print "Error: source from trace and from reference alignment should be the same"
    print s_id

  print>>out_file, src_trace
  print>>out_file, ' '.join(sentence_pair_trace.trg_tokens)
  print>>out_file, ' '.join(sentence_pair_ref.trg_tokens)
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    print>>out_file, vb.word, vb.posfine,
    for p in particles:
      print>>out_file, p,
    print>>out_file, ""
    for token in trace_tokens:
      print>>out_file, token,
    print>>out_file," : ",
    for pos,lemma in trace_tags:
      print>>out_file, lemma,
    print>>out_file, ""
    for token in ref_tokens:
      print>>out_file, token,
    print>>out_file," : ",
    for pos,lemma in ref_tags:
      print>>out_file, lemma,
    print>>out_file, ""
  print>>out_file, ""

matched_words = 0
trace_words = 0
ref_words = 0
stats = defaultdict(int)
ranks_trace = defaultdict(list)
ranks_ref = defaultdict(list)
ranks_trace_VN = defaultdict(list)
ranks_ref_VN = defaultdict(list)

def compute_accuracy(source_verbs):
  global stats #,matched_words, trace_words, ref_words
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    if len(ref_tokens) == 0:
      stats["no_aligned_ref_verb"] += 1
    for token, tags in zip(trace_tokens,trace_tags):
      if token in ref_tokens:
        stats["matched_words"] += 1
        if tags[0].startswith('V') or tags[0].startswith('N'):
          stats["matched_words_VN"] += 1
    for pos,lemma in trace_tags:
      if lemma in map(itemgetter(1),ref_tags):
        stats["matched_lemmas"] += 1
        if pos.startswith('V') or pos.startswith('N'):
          stats["matched_lemmas_VN"] += 1
    
    stats["trace_words"] += len(trace_tokens)
    stats["ref_words"] += len(ref_tokens)
    if len(trace_tokens)==0:
      stats["no_aligned_trace_verb"] += 1

def create_target_dict(target_options):
  target_dict = {}
  for item in target_options:
    for token in item:
      target_dict[token] += 1
  return target_dict

def find_token(token, target_options):
  indexes = []
  for index,item in enumerate(target_options):
    if token in item:
      indexes.append(index)
  return indexes

def search_VN_tokens_in_pt(all_tokens, all_tags, pt_trg_tokens_split, pt_entry, name, ref_tokens = None):
  top_ranks = []
  for token, tags in zip(all_tokens, all_tags):
    if tags[0].startswith('V') or tags[0].startswith('N'):
      indexes = find_token(token, pt_trg_tokens_split)
      if indexes:
        print name+" - found token: ",codecs.encode(token,'utf-8')
        stats[name+"_verb_token_in_pt"] += 1
        # compute ranks for verbs matching the reference (in case we have a reference) -> why only if they match the reference? -> so the counts are similar?
        if ref_tokens is None or (len(ref_tokens) >= 1 and token in ref_tokens):
          ranks = map(lambda x: pt_entry[x][3], indexes)
          ranks.sort()
          # lowest ranks -> rank[0]
          if ranks[0] == 1:
            stats[name+"_verb_token_in_pt_rank_1"] += 1
          if ranks[0] < 5:
            stats[name+"_verb_token_in_pt_rank_5"] += 1
          if ranks[0] < 10:
            stats[name+"_verb_token_in_pt_rank_10"] += 1
          # save top ranks
          top_ranks.append(ranks[0])
      else:
        print name+" not found token: ",codecs.encode(token,'utf-8')
        stats[name+"_verb_token_not_in_pt"] += 1
      stats["total_"+name+"_verb_token"] += 1
  return top_ranks


def find_in_pt(source_verbs, pt):
  global stats
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    if pt.has_key(vb.word):
      pt_entry = pt[vb.word]
      pt_trg_tokens =  map(itemgetter(0),pt_entry)
      pt_trg_tokens_split = map(lambda x: x.split(), pt_trg_tokens)
      print "\n", codecs.encode(vb.word,'utf-8')
      if len(ref_tokens) >= 1:
        found_ref = True
      ####### search for ref tokens in pt ##############
        top_ranks = search_VN_tokens_in_pt(ref_tokens, ref_tags, pt_trg_tokens_split, pt_entry, "ref")
        for rank in top_ranks:
          ranks_ref_VN[vb.word].append(rank)
        # find ref tokens
        ref = ' '.join(ref_tokens)
        if ref in pt_trg_tokens:
          print "Found ref: ",codecs.encode(ref,'utf-8')
          stats["ref_verb_in_pt"] += 1
          index = pt_trg_tokens.index(ref)
          # save ranks for each source verb
          ranks_ref[vb.word].append(pt_entry[index][3])
          # cummulate rank
          stats["ref_verb_in_pt_rank_total"] += pt_entry[index][3]
          # is the ref translation highest scoring?
          if pt_entry[index][3] == 1:
            stats["ref_verb_in_pt_rank_1"] += 1
        
        else:
          print "Not found ref: ",codecs.encode(ref,'utf-8')
          stats["ref_verb_not_in_pt"] += 1
    
      ####### search for trace tokens in pt ##############
      if len(trace_tokens) >= 1:
        # why compute the rank of trace verbs only if they match the reference?
        top_ranks = search_VN_tokens_in_pt(trace_tokens, trace_tags, pt_trg_tokens_split, pt_entry, "trace", ref_tokens)
        for rank in top_ranks:
          ranks_trace_VN[vb.word].append(rank)
        # find trace tokens
        trace = ' '.join(trace_tokens)
        # it should always be found in the pt used by the decoder
        #   -> the counter makes sense when comparing with other pt
        if trace in pt_trg_tokens:
          print "Found trace: ",codecs.encode(trace,'utf-8')
          stats["trace_verb_in_pt"] += 1
          index = pt_trg_tokens.index(trace)
          # save ranks for each source verb
          ranks_trace[vb.word].append(pt_entry[index][3])
          # cummulate rank
          stats["trace_verb_in_pt_rank_total"] += pt_entry[index][3]
          # is the trace translation highest scoring?
          if pt_entry[index][3] == 1:
            stats["trace_verb_in_pt_rank_1"] += 1
        else:
          print "Not found trace: ",codecs.encode(trace,'utf-8')
          stats["trace_verb_not_in_pt"] += 1
    else:
      stats["src_verb_not_in_pt"] += 1

#Source | target| scores | word alignment| count_target count_source count_joint | tree
#Scores:
# 1 - P(src|trg) = inverse phrase prob = count_joint/count_trg
# 3 - P(trg|src) = direct phrase prob = count_joint/count_src
# 2 - lex(src|trg) = inverse lexical weighting ?
# 4 - lex(trg|src) = direct  lexical weighting ?

def ReadPT(pt_file):
  pt = defaultdict(list)
  pt_line_re = re.compile(r"(.*) \[X\] \|\|\| (.*) \[(.*)\] \|\|\| (.*) \|\|\| (.*) \|\|\| (.*) \|\|\| \|\|\| (.*)")
  #^Trans Opt (\d+) \[(\d+)\.\.(\d+)\]: (.+)  : (\S+) \-\>\S+  \-\> (.+) :([\(\),\d\- ]*): term=(.*): nonterm=(.*): c=")
  for line in pt_file:
    match = pt_line_re.match(line)
    src = match.group(1).strip()
    trg = match.group(2).strip()
    trg_label = match.group(3)
    scores = map(float, match.group(4).split())
    #tokens = line.split("|||")
    #src = tokens[0].split("[")[0].strip()
    #trg = tokens[1].split("[")[0].strip()
    #trg_label = tokens[1].split("[")[1].split("]")[1]
    #scores = map(float,tokens[2].split())
    pt[src].append((trg, scores, trg_label))
  
  # rank entries by score: 3 - P(trg|src)
  ranked_pt = defaultdict(list)
  for src,entries in pt.items():
    # sorted(k['a'], key = lambda x: (x[1][0], x[1][1]))
    # sort by first score of each entry (trg,scores,trg_label)
    sorted_entries = sorted(entries, key = lambda x: (x[1][3]))
    # rank from highest to lowest and give equal values the min rank
    ranks = map(int,ss.rankdata(map(lambda x: 1-x, map(lambda y: y[1][3],sorted_entries)),method="min"))
    ranked_entry = [(item[0][0], item[0][1], item[0][2], item[1]) for item in zip(sorted_entries, ranks)]
    ranked_pt[src] = ranked_entry
  
  return ranked_pt

def median_rank_for_verb(verb,ranks):
  return numpy.median(ranks[verb])

def median_rank(ranks,sys):
  all_ranks = []
  median_per_verb = defaultdict(int)
  for key,item in ranks.items():
    all_ranks += item
    median_per_verb[key] = numpy.median(item)
  stats["median_rank_"+sys] = numpy.median(all_ranks)
  return numpy.median(all_ranks), median_per_verb


def print_stats(total_verbs):
  print "matched words against ref: ", stats["matched_words"]
  print "matched lemmas against ref: ", stats["matched_lemmas"]
  print "matched words (NV) against ref: ", stats["matched_words_VN"]
  print "matched lemmas (NV) against ref: ", stats["matched_lemmas_VN"]
  print "total source verbs:", total_verbs
  print "total ref verbs:", stats["ref_words"]
  print "total trace verbs:", stats["trace_words"]
  print ""
  
  print "R token = ", stats["matched_words"] *1.0/stats["ref_words"]
  print "P token = ", stats["matched_words"] *1.0/stats["trace_words"]
  print "R lemma = ", stats["matched_lemmas"] *1.0/stats["ref_words"]
  print "P lemma = ", stats["matched_lemmas"] *1.0/stats["trace_words"]
  print ""
  
  print "recall with respect to source verbs: "
  print "R token = ", stats["matched_words"] *1.0/total_verbs
  print "R lemma = ", stats["matched_lemmas"] *1.0/total_verbs
  print "R token (VN) = ", stats["matched_words_VN"] *1.0/total_verbs
  print "R lemma (VN) = ", stats["matched_lemmas_VN"] *1.0/total_verbs
  print ""

  print "ref verbs not aligned: ", stats["no_aligned_ref_verb"]
  print "trace verbs not aligned: ", stats["no_aligned_trace_verb"]
  print ""
  
  print "ref verbs found in pt: ", stats["ref_verb_in_pt"]
  print "ref verbs not found in pt: ", stats["ref_verb_not_in_pt"]
  print "trace verbs found in pt: ", stats["trace_verb_in_pt"]
  print "trace verbs not found in pt: ", stats["trace_verb_not_in_pt"]
  print "src verbs not found in pt: ", stats["src_verb_not_in_pt"]
  print "ref verb token in pt tokens", stats["ref_verb_token_in_pt"]
  print "ref verb token not in pt tokens", stats["ref_verb_token_not_in_pt"]
  print "total_ref_verb_token", stats["total_ref_verb_token"]
  print "trace verb token in pt tokens", stats["trace_verb_token_in_pt"]
  print "trace verb token not in pt tokens", stats["trace_verb_token_not_in_pt"]
  print "total_trace_verb_token", stats["total_trace_verb_token"]
  
  print ""
  print "average rank of ref verbs found in pt: ", stats["ref_verb_in_pt_rank_total"]*1.0/ stats["ref_verb_in_pt"]
  print "average rank of trace verbs found in pt: ", stats["trace_verb_in_pt_rank_total"]*1.0/ stats["trace_verb_in_pt"]
  # should divide by nr of source verbs to be comparable with the R token
  print "percentage ref verbs found in pt with rank 1: ", stats["ref_verb_in_pt_rank_1"]*1.0/stats["ref_verb_in_pt"]
  print "percentage trace verbs found in pt with rank 1: ", stats["trace_verb_in_pt_rank_1"]*1.0/ stats["trace_verb_in_pt"]
  print "percentage ref verbs tokens found in pt with rank 1: ", stats["ref_verb_token_in_pt_rank_1"]*1.0/ stats["ref_verb_token_in_pt"]
  print "percentage ref verbs tokens found in pt with rank <5: ", stats["ref_verb_token_in_pt_rank_5"]*1.0/ stats["ref_verb_token_in_pt"]
  print "percentage ref verbs tokens found in pt with rank <10: ", stats["ref_verb_token_in_pt_rank_10"]*1.0/ stats["ref_verb_token_in_pt"]
  print "percentage trace verbs tokens found in pt with rank 1: ", stats["trace_verb_token_in_pt_rank_1"]*1.0/ stats["trace_verb_token_in_pt"]
  print "percentage trace verbs tokens found in pt with rank <5: ", stats["trace_verb_token_in_pt_rank_5"]*1.0/ stats["trace_verb_token_in_pt"]
  print "percentage trace verbs tokens found in pt with rank <10: ", stats["trace_verb_token_in_pt_rank_10"]*1.0/ stats["trace_verb_token_in_pt"]
  print "median rank over all ref verbs in pt: ", stats["median_rank_ref"]
  print "median rank over all trace verbs in pt: ", stats["median_rank_trace"]
  print "median rank over all ref verbs tokens in pt: ", stats["median_rank_ref_VN"]
  print "median rank over all trace verbs tokens in pt: ", stats["median_rank_trace_VN"]

def main():
  global stats
  parser = argparse.ArgumentParser()
  parser.add_argument('--parse_file', default='/Subcat/parse_src.in',type=str)
  parser.add_argument('--trace_alignment', default='/Subcat/trace.out',type=str)
  parser.add_argument('--reference_alignment', default='/Subcat/trace.out',type=str)
  parser.add_argument('--out_verb_alignment', default='/Subcat/verb.out',type=str)
  parser.add_argument('--out_verbs_suffix', default ='/Subcat/verb.out',type=str)
  parser.add_argument('--pt_verbs', default ='/Subcat/EvalVerbs/phrase-table.src_verbs.100K.gz',type=str)
  parser.add_argument('--main_only_src', default = False, action='store_true')
  parser.add_argument('--main_only_trg', default = False, action='store_true')
  parser.add_argument('--Stanford_pos', default = False, action='store_true')
  args = parser.parse_args()

  fp = open(args.parse_file,'r')
  #codecs.open(args.parse_file, decoding='utf-8', mode = 'r', errors='ignore')
  ft = open(args.trace_alignment, 'r')
  #codecs.open(args.trace_alignment, encoding='utf-8', mode = 'r', errors='ignore')
  fr = open(args.reference_alignment, 'r')
  #codecs.open(args.reference_alignment, encoding='utf-8', mode = 'r', errors='ignore')
  fv = codecs.open(args.out_verb_alignment, encoding='utf-8', mode = 'w', errors='ignore')
  
  # output verb dictionaries
  f_srcv = codecs.open(args.out_verbs_suffix+".src", encoding='utf-8', mode = 'w', errors='ignore')
  f_tracev = codecs.open(args.out_verbs_suffix+".trace", encoding='utf-8', mode = 'w', errors='ignore')
  f_refv = codecs.open(args.out_verbs_suffix+".ref", encoding='utf-8', mode = 'w', errors='ignore')

  f_pt = codecs.getreader("utf-8")(gzip.open(args.pt_verbs, 'rt'))
  pt = ReadPT(f_pt)

  # initialize Stanford pos tagger
  st = None
  if args.Stanford_pos:
    from nltk.tag.stanford import StanfordPOSTagger
    from os.path import expanduser
    home = expanduser("/fs/maria/tools")
    _path_to_model = home + '/stanford-postagger/models/english-bidirectional-distsim.tagger'
    _path_to_jar = home + '/stanford-postagger/stanford-postagger.jar'
    st = StanfordPOSTagger(_path_to_model, _path_to_jar)

  s_id = 0
  total_verbs = 0
  while True:
    dep_graph = dp.CoNLLXFormat.read_depgraph(fp)
    if dep_graph == None:
      break
    #print dep_graph.words()
    sentence_pair_trace = SentencePair()
    sentence_pair_trace.read_sentence_pair(ft, st, False, False)
    
    sentence_pair_ref = SentencePair()
    sentence_pair_ref.read_sentence_pair(fr, st, False, False)

    source_verbs = SelectSourceVerbs.process_sentence_pair(args.main_only_src, args.main_only_trg, dep_graph, sentence_pair_trace, sentence_pair_ref)

    print_pair(source_verbs, sentence_pair_trace, sentence_pair_ref, fv, s_id)
    
    compute_accuracy(source_verbs)
    total_verbs += len(source_verbs)
  
    find_in_pt(source_verbs, pt)

    s_id += 1
    
  # compute median rank per verb and over all verbs
  median_rank_all_vb_ref, median_rank_per_verb_ref = median_rank(ranks_ref,"ref")
  median_rank_all_vb_trace, median_rank_per_verb_trace = median_rank(ranks_trace,"trace")
  median_rank_all_vb_ref_VN, median_rank_per_verb_ref_VN = median_rank(ranks_ref_VN,"ref_VN")
  median_rank_all_vb_trace_VN, median_rank_per_verb_trace_VN = median_rank(ranks_trace_VN,"trace_VN")

  # statistics - Precision, coverage, ranks
  print_stats(total_verbs)
  # print source verbs
  #print_dict_file2s(SelectSourceVerbs.source_verbs_types, f_srcv)
  print_combine_file2s(SelectSourceVerbs.source_verbs_types, median_rank_per_verb_ref, f_srcv)
  print_dict_file2s(SelectSourceVerbs.trace_verbs_types, f_tracev)
  print_dict_file2s(SelectSourceVerbs.ref_verbs_types, f_refv)

if __name__ == '__main__':
  
  main()



