# small extensions to these classes
# the original document was somewhere on the internet

import sys

# Represents a token in a dependency tree for a sentence.
class Token:

    # Makes a new token. If an array of strings is given, this is expected
    # to represent the columns in the CoNLL-X format.
    def __init__(self, arr=None):
        if arr:
            self.isdummy = False
            self.word = arr[1]
            self.pos = None if arr[3] == "_" else arr[3]
            self.lemma = None if arr[2] == "_" else arr[2]
            self.posfine = None if arr[4] == "_" else arr[4]
            self.feats = [] if arr[5] == "_" else arr[5].split(u"|")
            self.position = int(arr[0])
            self.arc_label = arr[7]
            self.head_position = int(arr[6])
        else:
            self.isdummy = True
            self.position = 0
            self.head = None
            self.arc_label = None
        self.dep_links = []
    
    # Makes a copy of this token. Does not copy the dependency arcs
    # going to and from this token.
    def clean_copy(self):
        out = Token()
        out.dep_links = []
        out.head = None
        out.arc_label = None
        if(not self.isdummy):
            out.isdummy = False
            out.word = self.word
            out.pos = self.pos
            out.lemma = self.lemma
            out.posfine = self.posfine
            out.feats = self.feats
            out.position = self.position
        return out

    # Connects this token as a dependent of a head token.
    # Will update the head and arc_label attributes, as well as
    # the dep_links attribute of the head token.
    def set_head(self, head, label=None):
        self.head = head
        self.arc_label = label
        if head:
            head.dep_links.append( (self, label) )

    # Returns the list of all links to dependent tokens.
    def get_dependents(self):
        return self.dep_links

    # Returns a list of dependent with a particular arc label.
    def get_dependents_by_label(self, label):
        out = []
        for (c, r) in self.dep_links:
            if r == label:
                out.append(c)
        return out

    # Returns the leftmost dependent of this token, i.e. the one
    # whose position attribute is the smallest.
    def get_leftmost_dependent(self):
        minpos = -1
        minc = None
        for (c, _) in self.dep_links:
            if minpos < 0 or c.position < minpos:
                minpos = c.position
                minc = c
        return minc

    # Returns the rightmost dependent of this token, i.e. the one
    # whose position attribute is the smallest.
    def get_rightmost_dependent(self):
        maxpos = -1
        maxc = None
        for (c, _) in self.dep_links:
            if maxpos < 0 or c.position > maxpos:
                maxpos = c.position
                maxc = c
        return maxc

    def __repr__(self):
        if self.isdummy:
            return "<0, <D>>"
        else:
            hs = (", " + str(self.head.position)) if self.head else ""
            return "<" + str(self.position) + ", " + ustr(self.word) + hs+ ">"

    def __str__(self):
        if self.isdummy:
            return "<0, <D>>"
        else:
            return "<" + str(self.position) + ", " + ustr(self.word) + ">"

class DependencyGraph:
    def __init__(self, tokens):
        self.tokens = tokens

    # Returns the length of the sentence, including the dummy root token.
    def length(self):
        return len(self.tokens)

    # Compares this parse tree to another parse tree for the same sentence.
    # Returns a tuple with these three numbers:
    # 1. number of correctly attached dependencies
    # 2. number of correctly attached and labeled dependencies
    # 3. total number of dependencies in the tree.
    def compare(self, other):
        if len(self.tokens) != len(other.tokens):
            raise ValueError("Sentences of different lengths")
        count_labeled = 0
        count_unlabeled = 0
        for i in range(1, len(self.tokens)):
            t1 = self.tokens[i]
            t2 = other.tokens[i]
            if t1.head.position == t2.head.position:
                count_unlabeled += 1      
                if t1.arc_label == t2.arc_label:
                    count_labeled += 1
        return (count_unlabeled, count_labeled, len(self.tokens) - 1)    

    # Returns a dependency tree containing the same tokens as this tree,
    # but with all dependency arcs removed.
    def clean_copy(self):
        return DependencyGraph(map(Token.clean_copy, self.tokens))

    def __str__(self):
        return "<DependencyGraph: " + str(self.tokens) + ">"

    def __repr__(self):
        return "<DependencyGraph: " + str(self.tokens) + ">"

    # Returns a string containing the words in the sentence.
    def words(self):
        out = ""
        for n in self.tokens:
            if not n.isdummy:
                out = out + n.word + u" "
        return out.strip()
        
    def words_list(self):
        out = []
        for n in self.tokens:
            if not n.isdummy:
                out.append(n.word)
        return out

    # Returns all tokens sorted by position
    def sorted_tokens(self):
        out = []
        for n in self.tokens:
            if not n.isdummy:
                out.append(n)
        return sorted(out, key = lambda x: x.position)


# This class contains functionality to read and write dependency trees
# encoded in the CoNLL-X format (http://ilk.uvt.nl/conll/#dataformat)
class CoNLLXFormat:    

    # Reads the line strings representing a dependency tree from the file f.
    @staticmethod
    def read_conllx_lines(f):
        line = f.readline().decode("utf-8")
        if not line:
            return None
        line = line.strip()
        if line == u"":
            return None
        lines = []
        while line and (line != u""):
            #print line
            lines.append(line)
            line = f.readline().decode("utf-8")
            if line:
                line = line.strip()
        return lines

    # Reads one dependency tree from the file f.
    @staticmethod
    def read_depgraph(f):
        lines = CoNLLXFormat.read_conllx_lines(f)
        if not lines:
            return None
        tokens = [Token(None)]
        for line in lines:
            tokens.append(Token(line.split("\t")))
        for token in tokens:
            if not token.isdummy:
                p = tokens[token.head_position]
                token.head = p
                p.dep_links.append((token, token.arc_label))
        return DependencyGraph(tokens)

    # CoNLL-X string representation of a list of morphological features.
    @staticmethod
    def feats_to_str(feats):
        if not feats or len(feats) == 0:
            return "_"
        return u"|".join(feats)

    # Writes one dependency tree to the file f.
    @staticmethod
    def write_depgraph(dg, f):
        for token in dg.tokens:
            if not token.isdummy:
                f.write(str(token.position))
                f.write("\t")
                f.write(ustr(token.word))
                f.write("\t")
                f.write(ustr(token.lemma) if token.lemma else "_")
                f.write("\t")
                f.write(ustr(token.pos))
                f.write("\t")
                f.write(ustr(token.posfine) if token.posfine else "_")
                f.write("\t")
                f.write(ustr(CoNLLXFormat.feats_to_str(token.feats)))
                f.write("\t")
                f.write(str(token.head.position) if token.head else "None")
                f.write("\t")
                f.write(ustr(token.arc_label) if token.arc_label else "_")
                f.write("\t_\t_\n")
        f.write("\n")

    # Reads all dependency trees in the file f, or up to maxlen trees
    # if this parameter is given.
    @staticmethod
    def read_treebank(f, maxlen=None):
        out = []
        tree = CoNLLXFormat.read_depgraph(f)
        while tree:
            out.append(tree)
            if maxlen and len(out) >= maxlen:
                break
            tree = CoNLLXFormat.read_depgraph(f)
        return out

    # Writes a list of dependency trees to the file f.
    @staticmethod
    def write_treebank(treebank, f):
        for dg in treebank:
            CoNLLXFormat.write_depgraph(dg, f)

# shorthand for UTF-8 encoding of a unicode, since UTF-8 is mandated
# by the CoNLL-X format.
def ustr(us):
    if not us:
        return u"None"
    else:
        return us.encode("utf-8")
