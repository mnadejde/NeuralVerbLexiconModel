# -*- coding: utf-8 -*-

# Util function for computing verb translation accuracy in different setups
# 1. Compute verb translation accuracy of 1-best translation as compared to the reference

import codecs
import argparse
from operator import itemgetter
import logging
from collections import defaultdict
import numpy

from Extract_verb_alignments import SentencePair
from Extract_verb_alignments import SelectSourceVerbs
import dependency as dp

def median_rank(ranks):
  all_ranks = []
  median_per_verb = defaultdict(int)
  for key,item in ranks.items():
    all_ranks += item
    median_per_verb[key] = numpy.median(item)
  return numpy.median(all_ranks), median_per_verb

def find_token(token, target_options):
  indexes = []
  for index,item in enumerate(target_options):
    if token in item:
      indexes.append(index)
  return indexes

def search_tokens_in_pt(tokens, pt_trg_tokens_split, pt_entry, name, cummulators):
  top_ranks = []
  for token in tokens:
    indexes = find_token(token, pt_trg_tokens_split)
    if indexes:
      print name+" - found token: ",codecs.encode(token,'utf-8')
      cummulators[name+"_verb_token_in_pt"] += 1
      # compute ranks for verbs matching the reference (in case we have a reference) -> why only if they match the reference? -> so the counts are similar?
      #if ref_tokens is None or (len(ref_tokens) >= 1 and token in ref_tokens):
      # assumes target options are sorted by P(t|s) scores and ranked
      ranks = map(lambda x: pt_entry[x][3], indexes)
      ranks.sort()
      # lowest ranks -> rank[0]
      if ranks[0] == 1:
        cummulators[name+"_verb_token_in_pt_rank_1"] += 1
      if ranks[0] < 5:
        cummulators[name+"_verb_token_in_pt_rank_5"] += 1
      if ranks[0] < 10:
        cummulators[name+"_verb_token_in_pt_rank_10"] += 1
      # save top ranks
      top_ranks.append(ranks[0])
    else:
      print name+" not found token: ",codecs.encode(token,'utf-8')
      cummulators[name+"_verb_token_not_in_pt"] += 1
    cummulators["total_"+name+"_verb_token"] += 1
  return top_ranks

# assume source verbs have only been select using main_only_src and main_only_trg -> no more prunning needed
def find_in_pt(source_verbs, pt, cummulators, ranks_ref, ranks_ref_tokens, ranks_trace, ranks_trace_tokens):
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    cummulators["source_verbs"] +=1
    if len(ref_tokens) == 0:
      cummulators["unaligned_ref"] += 1
    else:
      cummulators["reference_verbs"] += 1
    if len(trace_tokens) == 0:
      cummulators["unaligned_translation"] +=1
    else:
      cummulators["translation_verbs"] +=1
    # should be at most one translation verb token and one reference verb token -> constraints in process_sentence_pair
    print vb.word.encode("utf-8")
    print trace_tokens
    print ref_tokens
    
    # pt(src_verb)= [(trg_string, scores, trg_label, rank)] -> for the same source and target we can have different POS label and scores
    if pt.has_key(vb.word):
      pt_entry = pt[vb.word]
      pt_trg_tokens =  map(itemgetter(0),pt_entry)
      pt_trg_tokens_split = map(lambda x: x.split(), pt_trg_tokens)
      if len(ref_tokens) >= 1:
        found_ref = True
      ####### search for ref tokens in pt ##############
        top_ranks = search_tokens_in_pt(ref_tokens, pt_trg_tokens_split, pt_entry, "ref", cummulators)
        # ref_tokens should have only one target verb and therefore top_ranks should only have one element
        for rank in top_ranks:
          ranks_ref_tokens[vb.word].append(rank)
        # find ref tokens
        ref = ' '.join(ref_tokens)
        if ref in pt_trg_tokens:
          print "Found ref: ",codecs.encode(ref,'utf-8')
          cummulators["ref_verb_in_pt"] += 1
          index = pt_trg_tokens.index(ref)
          # save ranks for each source verb
          ranks_ref[vb.word].append(pt_entry[index][3])
          # is the ref translation highest scoring?
          if pt_entry[index][3] == 1:
            cummulators["ref_verb_in_pt_rank_1"] += 1
          if pt_entry[index][3] < 5:
            cummulators["ref_verb_in_pt_rank_5"] += 1
          if pt_entry[index][3] < 10:
            cummulators["ref_verb_in_pt_rank_10"] += 1
        
        else:
          print "Not found ref: ",codecs.encode(ref,'utf-8')
          cummulators["ref_verb_not_in_pt"] += 1
    
      ####### search for trace tokens in pt ##############
      if len(trace_tokens) >= 1:
        # why compute the rank of trace verbs only if they match the reference?
        top_ranks = search_tokens_in_pt(trace_tokens, pt_trg_tokens_split, pt_entry, "trace", cummulators)
        for rank in top_ranks:
          ranks_trace_tokens[vb.word].append(rank)
        # find trace tokens
        trace = ' '.join(trace_tokens)
        # it should always be found in the pt used by the decoder
        #   -> the counter makes sense when comparing with other pt
        if trace in pt_trg_tokens:
          print "Found trace: ",codecs.encode(trace,'utf-8')
          cummulators["trace_verb_in_pt"] += 1
          index = pt_trg_tokens.index(trace)
          # save ranks for each source verb
          ranks_trace[vb.word].append(pt_entry[index][3])
           # is the trace translation highest scoring?
          if pt_entry[index][3] == 1:
            cummulators["trace_verb_in_pt_rank_1"] += 1
          if pt_entry[index][3] < 5:
            cummulators["trace_verb_in_pt_rank_5"] += 1
          if pt_entry[index][3] < 10:
            cummulators["trace_verb_in_pt_rank_10"] += 1
        else:
          print "Not found trace: ",codecs.encode(trace,'utf-8')
          cummulators["trace_verb_not_in_pt"] += 1
    else:
      cummulators["src_verb_not_in_pt"] += 1

def report_pt_stats(cummulators, ranks_ref, ranks_ref_tokens, ranks_trace, ranks_trace_tokens, f_stats):
  median_ref, median_ref_per_verb = median_rank(ranks_ref)
  median_ref_tokens, median_ref_rokens_per_verb = median_rank(ranks_ref_tokens)
  median_trace, median_trace_per_verb = median_rank(ranks_trace)
  median_trace_tokens, median_trace_tokens_per_verb = median_rank(ranks_trace_tokens)
  
  print>>f_stats, "total source verbs: ", cummulators["source_verbs"]
  print>>f_stats, "total translation verbs: ", cummulators["translation_verbs"]
  print>>f_stats, "total reference verbs: ", cummulators["reference_verbs"]
  print>>f_stats, "total unaligned translation verbs: ", cummulators["unaligned_translation"]
  print>>f_stats, "total unaligned reference verbs: ", cummulators["unaligned_ref"]
  
  print>>f_stats, "\n\nREFERENCE VERBS\n\n"
  
  # report ref verbs found in pt
  print>>f_stats, "ref_verb_token_in_pt: ", cummulators["ref_verb_token_in_pt"], cummulators["ref_verb_token_in_pt"] *1.0 / cummulators["reference_verbs"] #cummulators["source_verbs"]
  print>>f_stats, "ref_verb_in_pt: ", cummulators["ref_verb_in_pt"], cummulators["ref_verb_in_pt"] *1.0 / cummulators["reference_verbs"] #cummulators["source_verbs"]
  
  print>>f_stats, "ref_verb_token_not_in_pt: ", cummulators["ref_verb_token_not_in_pt"], cummulators["ref_verb_token_not_in_pt"] *1.0 / cummulators["reference_verbs"] #cummulators["source_verbs"]
  print>>f_stats, "ref_verb_not_in_pt: ", cummulators["ref_verb_not_in_pt"], cummulators["ref_verb_not_in_pt"] *1.0 / cummulators["reference_verbs"] #cummulators["source_verbs"]

  # report ranks for ref verbs in pt
  print>>f_stats, "ref_verb_median_rank:", median_ref
  print>>f_stats, "ref_verb_tokens_median_rank:", median_ref_tokens
  print>>f_stats, "ref_verb_token_in_pt_rank_1: ", cummulators["ref_verb_token_in_pt_rank_1"], cummulators["ref_verb_token_in_pt_rank_1"] *1.0 / cummulators["ref_verb_token_in_pt"]
  print>>f_stats, "ref_verb_token_in_pt_rank_5: ", cummulators["ref_verb_token_in_pt_rank_5"], cummulators["ref_verb_token_in_pt_rank_5"] *1.0 / cummulators["ref_verb_token_in_pt"]
  print>>f_stats, "ref_verb_token_in_pt_rank_10: ", cummulators["ref_verb_token_in_pt_rank_10"], cummulators["ref_verb_token_in_pt_rank_10"] *1.0 / cummulators["ref_verb_token_in_pt"]

  print>>f_stats, "\n\nTRANSLATION VERBS\n\n"
  
  # report trace verbs found in pt -> not directly comparable with ref stats because denominator is different
  print>>f_stats, "trace_verb_token_in_pt: ", cummulators["trace_verb_token_in_pt"], cummulators["trace_verb_token_in_pt"] *1.0 / cummulators["translation_verbs"] #cummulators["source_verbs"]
  print>>f_stats, "trace_verb_in_pt: ", cummulators["trace_verb_in_pt"], cummulators["trace_verb_in_pt"] *1.0 / cummulators["translation_verbs"] #cummulators["source_verbs"]
  
  print>>f_stats, "trace_verb_token_not_in_pt: ", cummulators["trace_verb_token_not_in_pt"], cummulators["trace_verb_token_not_in_pt"] *1.0 / cummulators["translation_verbs"] #cummulators["source_verbs"]
  print>>f_stats, "trace_verb_not_in_pt: ", cummulators["trace_verb_not_in_pt"], cummulators["trace_verb_not_in_pt"] *1.0 / cummulators["translation_verbs"] #cummulators["source_verbs"]

  # report ranks for trace verbs in pt
  print>>f_stats, "trace_verb_median_rank:", median_trace
  print>>f_stats, "trace_verb_tokens_median_rank:", median_trace_tokens
  print>>f_stats, "trace_verb_token_in_pt_rank_1: ", cummulators["trace_verb_token_in_pt_rank_1"], cummulators["trace_verb_token_in_pt_rank_1"] *1.0 / cummulators["trace_verb_token_in_pt"]
  print>>f_stats, "trace_verb_token_in_pt_rank_5: ", cummulators["trace_verb_token_in_pt_rank_5"], cummulators["trace_verb_token_in_pt_rank_5"] *1.0 / cummulators["trace_verb_token_in_pt"]
  print>>f_stats, "trace_verb_token_in_pt_rank_10: ", cummulators["trace_verb_token_in_pt_rank_10"], cummulators["trace_verb_token_in_pt_rank_10"] *1.0 / cummulators["trace_verb_token_in_pt"]



# input:
#     processed SentencePair -> SelectSourceVerbs.process_sentence_pair
#     cummulators -> dictionary of statistics to be updated using the current example
# assumptions:
#     SentencePair processed source, translation, reference and selected only the main source/target verb token
#     statistics will be computed by comparing with the main target verb token only
def compute_translation_accuracy(source_verbs, cummulators):
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    cummulators["source_verbs"] +=1
    if particles:
      cummulators["source_particle_verbs"] +=1
    if len(ref_tokens) == 0:
      cummulators["unaligned_ref"] += 1
    else:
      cummulators["reference_verbs"] += 1
      if particles:
        cummulators["ref_particle_verbs"] +=1
        if len(trace_tokens) != 0:
				  cummulators["ref_translation_particle_verbs"] += 1
      if len(trace_tokens) != 0:
        cummulators["ref_translation_verbs"] += 1
    if len(trace_tokens) == 0:
      cummulators["unaligned_translation"] +=1
    else:
      cummulators["translation_verbs"] +=1
    # should be at most one translation verb token and one reference verb token -> constraints in process_sentence_pair
    print vb.word.encode("utf-8")
    print trace_tokens
    print ref_tokens
    for token in trace_tokens:
      if token in ref_tokens:
        cummulators["matched_words"] +=1
        if particles:
          cummulators["matched_particle_words"] +=1
        print "matched"
    # should have provided pos and lemmas, otherwise this will give error
    for pos, lemma in trace_tags:
      print lemma.encode("utf-8")
      if lemma in map(itemgetter(1),ref_tags):
        cummulators["matched_lemmas"] +=1
        if particles:
          cummulators["matched_particle_lemmas"] +=1
        print "matched lemma"
  print ""

def compute_translation_accuracy_for_nbest(source_verbs, cummulators, matched_verbs, matched_verb_lemmas, seen_verbs):
  for vb, trace_tokens, ref_tokens, particles, trace_tags, ref_tags in source_verbs:
    # only process source verbs for which a match has not been found between nbest translation and ref translation
    if matched_verbs.has_key(vb) and matched_verb_lemmas.has_key(vb):
      continue
    # count source and reference verb stats only once per source verb, not for every nbest entry
    if not seen_verbs.has_key(vb):
      seen_verbs[vb] += 1
      cummulators["source_verbs"] +=1
      if len(ref_tokens) == 0:
        cummulators["unaligned_ref"] += 1
      else:
        cummulators["reference_verbs"] += 1
    if len(trace_tokens) == 0:
      cummulators["unaligned_translation"] +=1
    else:
      cummulators["translation_verbs"] +=1
    # should be at most one translation verb token and one reference verb token -> constraints in process_sentence_pair
    #print vb.word.encode("utf-8")
    #print trace_tokens
    #print ref_tokens
    if not matched_verbs.has_key(vb):
      for token in trace_tokens:
        if token in ref_tokens:
          cummulators["matched_words"] +=1
          matched_verbs[vb] +=1
          #print "matched"
    if not matched_verb_lemmas.has_key(vb):
      for pos, lemma in trace_tags:
        #print lemma.encode("utf-8")
        if lemma in map(itemgetter(1),ref_tags):
          cummulators["matched_lemmas"] +=1
          matched_verb_lemmas[vb] +=1
          #print "matched lemma"
  #print ""

def test_compute_translation_accuracy():
  return

def report_translation_accuracy_stats(cummulators, f_stats):
  verb_type_translation_accuracy = cummulators["matched_words"] * 1.0/ cummulators["reference_verbs"] #cummulators["source_verbs"]
  verb_lemma_translation_accuracy = cummulators["matched_lemmas"] * 1.0 / cummulators["reference_verbs"] #cummulators["source_verbs"]
	# look at translation precision for verbs that have both an aligned translation and an aligned reference
	# it's different from the 1-best evaluation since different hypothesis have different number of verbs
  verb_type_translation_precision = cummulators["matched_words"] * 1.0/ cummulators["ref_translation_verbs"]
  verb_lemma_translation_precision = cummulators["matched_lemmas"] * 1.0 / cummulators["ref_translation_verbs"]
  verb_type_translation_F1 = 2.0 * verb_type_translation_precision * verb_type_translation_accuracy / (verb_type_translation_precision + verb_type_translation_accuracy)
  verb_lemma_translation_F1 = 2.0 * verb_lemma_translation_precision * verb_lemma_translation_accuracy / (verb_lemma_translation_precision + verb_lemma_translation_accuracy)
	
	#recall?
  particle_verb_type_translation_accuracy = cummulators["matched_particle_words"] *1.0 / cummulators["ref_particle_verbs"]
  particle_verb_lemma_translation_accuracy = cummulators["matched_particle_lemmas"] *1.0 / cummulators["ref_particle_verbs"]
	#precision
  particle_verb_type_translation_precision = cummulators["matched_particle_words"] *1.0 / cummulators["ref_translation_particle_verbs"]
  particle_verb_lemma_translation_precision = cummulators["matched_particle_lemmas"] *1.0 / cummulators["ref_translation_particle_verbs"]
  particle_verb_type_translation_F1 = 2.0 * particle_verb_type_translation_precision * particle_verb_type_translation_accuracy / (particle_verb_type_translation_precision + particle_verb_type_translation_accuracy)
  particle_verb_lemma_translation_F1 = 2.0 * particle_verb_lemma_translation_precision * particle_verb_lemma_translation_accuracy / (particle_verb_lemma_translation_precision + particle_verb_lemma_translation_accuracy)
  
  print>>f_stats, "total source verbs: ", cummulators["source_verbs"]
  print>>f_stats, "total translation verbs: ", cummulators["translation_verbs"]
  print>>f_stats, "total reference verbs: ", cummulators["reference_verbs"]
  print>>f_stats, "total aligned both reference & translation verbs: ", cummulators["ref_translation_verbs"], cummulators["reference_verbs"] - cummulators["ref_translation_verbs"]
  print>>f_stats, "total particle reference verbs: ", cummulators["ref_particle_verbs"]
  print>>f_stats, "total aligned both reference & translation particle verbs: ", cummulators["ref_translation_particle_verbs"], cummulators["ref_particle_verbs"] - cummulators["ref_translation_particle_verbs"]
  print>>f_stats, "total unaligned translation verbs: ", cummulators["unaligned_translation"]
  print>>f_stats, "total unaligned reference verbs: ", cummulators["unaligned_ref"]
  print>>f_stats, "verb type translation accuracy: ", verb_type_translation_accuracy
  print>>f_stats, "verb lemma translation accuracy: ", verb_lemma_translation_accuracy
  print>>f_stats, "verb type translation precision: ", verb_type_translation_precision
  print>>f_stats, "verb lemma translation precision: ", verb_lemma_translation_precision
  print>>f_stats, "verb type translation F1: ", verb_type_translation_F1
  print>>f_stats, "verb lemma translation F1: ", verb_lemma_translation_F1
  print>>f_stats, "particle verb type translation accuracy (recall): ", particle_verb_type_translation_accuracy
  print>>f_stats, "particle verb lemma translation accuracy (recall): ", particle_verb_lemma_translation_accuracy
  print>>f_stats, "particle verb type translation precision : ", particle_verb_type_translation_precision
  print>>f_stats, "particle verb lemma translation precision: ", particle_verb_lemma_translation_precision
  print>>f_stats, "particle verb type translation F1: ", particle_verb_type_translation_F1
  print>>f_stats, "particle verb lemma translation F1: ", particle_verb_lemma_translation_F1



def read_next_input(f_parse, f_translation, f_reference):
  # read dependency tree in CONLL format for each sentence
  dep_graph = dp.CoNLLXFormat.read_depgraph(f_parse)
  if dep_graph == None:
    return None, None, None

  # read sentence pairs -> for each sentence 4 lines: source sentence, target sentence, target pos, target lemmas, align
  sentence_pair_translation = SentencePair()
  sentence_pair_translation.read_sentence_pair(f_translation, False, True, True)
  #sentence_pair_translation.print_all_aligned_words()

  sentence_pair_reference = SentencePair()
  sentence_pair_reference.read_sentence_pair(f_reference, False, True, True)
  #sentence_pair_reference.print_all_aligned_words()

  return dep_graph, sentence_pair_translation, sentence_pair_reference

def read_next_input_from_nbest(f_parse, f_source, f_nbest, f_nbest_pos, f_nbest_lemma, f_reference, sid):
  # read dependency tree in CONLL format for each sentence
  dep_graph = dp.CoNLLXFormat.read_depgraph(f_parse)
  if dep_graph == None:
    return None, None, None

  sentence_pairs_nbest = SentencePair.read_nbest_separate(f_source, f_nbest, f_nbest_pos, f_nbest_lemma, sid)

  sentence_pair_reference = SentencePair()
  sentence_pair_reference.read_sentence_pair(f_reference, False, True, True)
  #sentence_pair_reference.print_all_aligned_words()

  return dep_graph, sentence_pairs_nbest, sentence_pair_reference

LOG = logging.getLogger(__name__)

# 1. Compute verb translation accuracy of 1-best translation as compared to the reference
# For both translation and reference files we have as input source sentence, target sentence, target pos, target lemmas
# We consider only main verbs on the source side which are identfied using the dependency parse
# Among the target tokens aligned to the source verb we try to select the main verb (VV*) or any other verb based on TreeTagger POS tags
def main():
  logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument('--in_parse_file_name',type=str)
  parser.add_argument('--in_translation_file_name', type=str)
  parser.add_argument('--in_reference_file_name', type=str)
  parser.add_argument('--out_stats_file_name', type=str)
  
  parser.set_defaults(
    in_parse_file_name='/PredictVerbs/newstest201345.input.split.9.parse.test', #test.nbest.parse'
    in_translation_file_name='/Subcat/PredictVerbs/newstest2013.reference.tc.9.align_tags',
    in_reference_file_name='/Subcat/PredictVerbs/newstest2013.reference.tc.9.align_tags2',
    out_stats_file_name='/Subcat/PredictVerbs/newstest2013.stats'
  )
  args = parser.parse_args()
  
  f_parse = open(args.in_parse_file_name,'r')
  f_translation = open(args.in_translation_file_name, 'r')
  #codecs.open(args.trace_alignment, encoding='utf-8', mode = 'r', errors='ignore')
  f_reference = open(args.in_reference_file_name, 'r')
  f_stats = open(args.out_stats_file_name, 'w')
  
  args = parser.parse_args()
  LOG.info("Arguments: %s" % args)

  cummulators = defaultdict(int)

  while True:
    # for each input dependency parse and 4 lines: source sentence, target sentence, target pos, target lemmas
    dep_graph, sentence_pair_translation, sentence_pair_reference = read_next_input(f_parse, f_translation, f_reference)
    if dep_graph == None:
      break
    # get word alignments for source verbs with corresponding pos and lemma tags
    main_only_src = True
    main_only_trg = True
    source_verbs = SelectSourceVerbs.process_sentence_pair(main_only_src, main_only_trg, dep_graph, sentence_pair_translation, sentence_pair_reference)
    # aggregate scores for matching verbs
    compute_translation_accuracy(source_verbs, cummulators)

  report_translation_accuracy_stats(cummulators, f_stats)

if __name__ == '__main__':
  
  main()
