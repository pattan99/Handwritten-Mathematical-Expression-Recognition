import string
import pandas as pd
import re


# string.printable[:95]
# '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
class VectorizeChar:
    def __init__(self, max_len=100):
        
        operational_tokens = [r'\blank',r'\sos', r'\eos']

        syntactic_tokens = [r"_", r"^", r'{', r'}', r'&', r'\\', ' ']
        #print(syntactic_tokens)
        latin_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        #print(latin_letters)
        numbers = [str(i) for i in range(10)]
        #print(numbers)
        blackboard = [r'\mathbb'] + ['\\mathbb{'+ chr(x) + '}' for x in range(ord('A'), ord('Z')+1)]
        #print(blackboard)
        latin_punctuations_symbols = [',', ';', ':', '!', '?', '.', '(', ')', '[', ']', r'\{', r'\}', '*', '/', '+', '-', r'\_', r'\&', r'\#', r'\%', '|', r'\backslash']
        #print(latin_punctuations_symbols)
        greek_letters = r'\alpha \beta \delta \Delta \epsilon \eta \chi \gamma \Gamma \iota \kappa \lambda \Lambda \nu \mu \omega \Omega \phi \Phi \pi \Pi \psi \Psi \rho \sigma \Sigma \tau \theta \Theta \upsilon \Upsilon \varphi \varpi \varsigma \vartheta \xi \Xi \zeta'
        greek_letters = greek_letters.split(' ')
        #print(greek_letters)
        math_constructs = r'\frac \sqrt \prod \sum \iint \int \oint'.split(' ')
        #print(math_constructs)
        modifiers = r'\hat \tilde \vec \overline \underline \prime \dot \not'.split(' ')
        #print(modifiers)
        matrix_env = r'\begin{matrix} \end{matrix}'.split(' ')
        # print(matrix_env)
        delimiters = r'\langle \rangle \lceil \rceil \lfloor \rfloor \|'.split(' ')
        #print(delimiters)
        conparisons = [r'\ge', r'\gg', r'\le', r'\ll', '<', '>']
        #print(conparisons)
        eq_aprox = r'= \approx \cong \equiv \ne \propto \sim \simeq'.split(' ')
        #print(eq_aprox)
        set_theory = r'\in \ni \notin \sqsubseteq \subset \subseteq \subsetneq \supset \supseteq \emptyset'.split(' ')
        #print(set_theory)
        operators = r'\times \bigcap \bigcirc \bigcup \bigoplus \bigvee \bigwedge \cap \cup \div \mp \odot \ominus \oplus \otimes \pm \vee \wedge'.split(' ')
        #print(operators)
        arrows = r'\hookrightarrow \leftarrow \leftrightarrow \Leftrightarrow \longrightarrow \mapsto \rightarrow \Rightarrow \rightleftharpoons \iff'.split(' ')
        #print(arrows)
        dots = r'\bullet \cdot \circ'.split(' ')
        #print(dots)
        others = r'\aleph \angle \dagger \exists \forall \hbar \infty \models \nabla \neg \partial \perp \top \triangle \triangleleft \triangleq \vdash \Vdash \vdots'.split(' ')
        #print(others)
        #print('\n\n')

        total = operational_tokens + syntactic_tokens + latin_letters + numbers + blackboard + latin_punctuations_symbols + greek_letters + math_constructs + modifiers + matrix_env + delimiters + conparisons + eq_aprox + set_theory + operators + arrows + dots + others
        # print(len(total))
        # print(total)
        self.vocab = total
        self.max_len = len(total)
        self.char_to_idx = {}
        self._COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def tokenize_expression(self, s: str) -> list[str]:
        r"""Transform a Latex math string into a list of tokens.

        Tokens are strings that are meaningful in the context of Latex
        e.g. '1', r'\alpha', r'\frac'.

        Args:
            s: unicode input string (ex: r"\frac{1}{2}")

        Returns:
            tokens: list of tokens as unicode strings.
        """
        tokens = []
        while s:
            if s[0] == '\\':
                tokens.append(self._COMMAND_RE.match(s).group(0))
            else:
                tokens.append(s[0])

            s = s[len(tokens[-1]) :]

        return tokens

    def __call__(self, text):
        # text = text[: self.max_len - 2]
        # text = r"\sos" + text + "\eos"
        text = self.tokenize_expression(text)
        pad_len = self.max_len - len(text)
        result = [1] + [self.char_to_idx.get(ch, 1) for ch in text] + [2]
        return  result#+ [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

#  for evaluation

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
    
class VectorizeCharCLIP:
    def __init__(self, max_len=100):
        operational_tokens = [r'\blank',r'\sos', r'\eos']

        syntactic_tokens = [r"_", r"^", r'{', r'}', r'&', r'\\', ' ']
        #print(syntactic_tokens)
        latin_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        #print(latin_letters)
        numbers = [str(i) for i in range(10)]
        #print(numbers)
        blackboard = [r'\mathbb'] + ['\\mathbb{'+ chr(x) + '}' for x in range(ord('A'), ord('Z')+1)]
        #print(blackboard)
        latin_punctuations_symbols = [',', ';', ':', '!', '?', '.', '(', ')', '[', ']', r'\{', r'\}', '*', '/', '+', '-', r'\_', r'\&', r'\#', r'\%', '|', r'\backslash']
        #print(latin_punctuations_symbols)
        greek_letters = r'\alpha \beta \delta \Delta \epsilon \eta \chi \gamma \Gamma \iota \kappa \lambda \Lambda \nu \mu \omega \Omega \phi \Phi \pi \Pi \psi \Psi \rho \sigma \Sigma \tau \theta \Theta \upsilon \Upsilon \varphi \varpi \varsigma \vartheta \xi \Xi \zeta'
        greek_letters = greek_letters.split(' ')
        #print(greek_letters)
        math_constructs = r'\frac \sqrt \prod \sum \iint \int \oint'.split(' ')
        #print(math_constructs)
        modifiers = r'\hat \tilde \vec \overline \underline \prime \dot \not'.split(' ')
        #print(modifiers)
        matrix_env = r'\begin{matrix} \end{matrix}'.split(' ')
        # print(matrix_env)
        delimiters = r'\langle \rangle \lceil \rceil \lfloor \rfloor \|'.split(' ')
        #print(delimiters)
        conparisons = [r'\ge', r'\gg', r'\le', r'\ll', '<', '>']
        #print(conparisons)
        eq_aprox = r'= \approx \cong \equiv \ne \propto \sim \simeq'.split(' ')
        #print(eq_aprox)
        set_theory = r'\in \ni \notin \sqsubseteq \subset \subseteq \subsetneq \supset \supseteq \emptyset'.split(' ')
        #print(set_theory)
        operators = r'\times \bigcap \bigcirc \bigcup \bigoplus \bigvee \bigwedge \cap \cup \div \mp \odot \ominus \oplus \otimes \pm \vee \wedge'.split(' ')
        #print(operators)
        arrows = r'\hookrightarrow \leftarrow \leftrightarrow \Leftrightarrow \longrightarrow \mapsto \rightarrow \Rightarrow \rightleftharpoons \iff'.split(' ')
        #print(arrows)
        dots = r'\bullet \cdot \circ'.split(' ')
        #print(dots)
        others = r'\aleph \angle \dagger \exists \forall \hbar \infty \models \nabla \neg \partial \perp \top \triangle \triangleleft \triangleq \vdash \Vdash \vdots'.split(' ')
        #print(others)
        #print('\n\n')

        total = operational_tokens + syntactic_tokens + latin_letters + numbers + blackboard + latin_punctuations_symbols + greek_letters + math_constructs + modifiers + matrix_env + delimiters + conparisons + eq_aprox + set_theory + operators + arrows + dots + others
        print(f"Vocab size: {len(total)}")
        self.vocab = total
        self.max_len = max_len
        self.char_to_idx = {}
        self._COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def tokenize_expression(self, s: str) -> list[str]:
        r"""Transform a Latex math string into a list of tokens.

        Tokens are strings that are meaningful in the context of Latex
        e.g. '1', r'\alpha', r'\frac'.

        Args:
            s: unicode input string (ex: r"\frac{1}{2}")

        Returns:
            tokens: list of tokens as unicode strings.
        """
        tokens = []
        while s:
            if s[0] == '\\':
                tokens.append(self._COMMAND_RE.match(s).group(0))
            else:
                tokens.append(s[0])

            s = s[len(tokens[-1]) :]

        return tokens

    def __call__(self, text):
        text = self.tokenize_expression(text)
        pad_len = self.max_len - len(text)
        result = [1] + [self.char_to_idx.get(ch, 1) for ch in text] + [2]
        return  result + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab