import subprocess
import re

result = subprocess.run([
    "pandoc", "paper.md",
    "--bibliography", "paper.bib",
    "--citeproc",
    "--csl", "numeric.csl",
    "-t", "latex"
], capture_output=True, text=True, encoding="utf-8", errors="replace")

body = result.stdout

# Clean up pandoc's References section wrapper
body = re.sub(
    r'\\hypertarget\{references\}\{%\s*\\section\*\{References\}\\label\{references\}\}\s*\\addcontentsline\{toc\}\{section\}\{References\}',
    r'\\section*{References}',
    body
)
body = re.sub(r'\\hypertarget\{refs\}\{\}\s*', '', body)

doc = r"""\documentclass[12pt, letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!70!black,citecolor=blue!70!black,urlcolor=blue!70!black]{hyperref}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{orcidlink}
\usepackage{booktabs}
\usepackage{graphicx}

\setstretch{1.15}
\setlength{\parskip}{6pt}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\vspace{2pt}\hrule height 0.8pt\vspace{2pt}]
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\titlespacing{\section}{0pt}{18pt}{8pt}
\titlespacing{\subsection}{0pt}{12pt}{4pt}

\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\textit{SAMA: Open-Source Microgrid Optimization Platform}}
\fancyhead[R]{\small\textit{Sadat \& Pearce, 2026}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0pt}

% CSLReferences: paragraph-based, no list, hanging indent
\newlength{\cslhangindent}
\setlength{\cslhangindent}{2.5em}
\newenvironment{CSLReferences}[2]{%
  \setlength{\parindent}{0pt}%
  \setlength{\leftskip}{\cslhangindent}%
  \everypar{\setlength{\hangindent}{\cslhangindent}}%
  \small%
  \setlength{\parskip}{4pt}%
}{\par\setlength{\leftskip}{0pt}\setlength{\parskip}{6pt}}

% CSLLeftMargin + CSLRightInline render inline as "label text"
\newcommand{\CSLBlock}[1]{#1}
\newcommand{\CSLLeftMargin}[1]{#1}
\newcommand{\CSLRightInline}[1]{#1}
\newcommand{\CSLIndent}[1]{\hspace{\cslhangindent}#1}
\newcommand{\citeproctext}{}
\newcommand{\citeproc}[2]{#2}

\begin{document}

\begin{center}
{\LARGE\bfseries
SAMA: Open-Source Multi-Objective Optimization Platform\\[4pt]
for Hybrid Renewable Energy Microgrid Design}\\[8pt]
\rule{\linewidth}{1.5pt}\\[10pt]
{\normalsize
  \textbf{Seyyed Ali Sadat}\textsuperscript{1}~\orcidlink{0000-0001-9690-4239}\quad
  \textbf{Joshua M. Pearce}\textsuperscript{1}~\orcidlink{0000-0001-9802-3056}
}\\[5pt]
{\small\textit{%
  \textsuperscript{1}Department of Electrical \& Computer Engineering,
  Western University, London, ON N6A 3K7, Canada
}}\\[4pt]
{\small May 7, 2026}\\[6pt]
\rule{\linewidth}{0.6pt}
\end{center}
\thispagestyle{fancy}
\vspace{4pt}

""" + body + r"""
\end{document}
"""

with open("paper_built.tex", "w", encoding="utf-8") as f:
    f.write(doc)

for i in range(2):
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "paper_built.tex"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

print("Done - paper_built.pdf is ready")
