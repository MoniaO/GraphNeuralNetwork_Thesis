%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bachelor's & Master's Thesis Template             %%
%% Copyleft by Artur M. Brodzki & Piotr Woźniak      %%
%% Faculty of Electronics and Information Technology %%
%% Warsaw University of Technology, 2019-2020        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ProvidesClass{szablon/eiti/eiti-thesis}

\LoadClass[
    12pt,
    twoside
]{mwart}

%--------------------------------------
% Define class parameters
%--------------------------------------
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family=szablon/eiti/eiti-thesis.cls,
    prefix=eiti
}

\DeclareStringOption[2.5cm]{left}
\DeclareStringOption[2.5cm]{right}
\DeclareStringOption[2.5cm]{top}
\DeclareStringOption[2.5cm]{bottom}
\DeclareStringOption[6mm]{bindingoffset}
\DeclareBoolOption[false]{nohyphenation}
\ProcessKeyvalOptions*

%--------------------------------
% Common packages
%--------------------------------
\RequirePackage{amsmath}    % Basic mathematical typesetting
\RequirePackage{amssymb}    % Advanced math symbols
\RequirePackage{amsthm}     % Theorems typesetting
\RequirePackage{array}      % Advanced table column formats
\RequirePackage[            % Bibliography in biber
    backend=biber,
    style=ieee
]{biblatex}
\RequirePackage[            % Custom figure and table captions
    font=small,
    labelfont=bf,
    labelsep=period
]{caption}
\RequirePackage{chngcntr}   % Per-section counters
\RequirePackage{enumitem}   % Itemize/enumrate
\RequirePackage{fancyhdr}   % Custom header/footer styles
\RequirePackage{fourier}    % Adobe Utopia font
\RequirePackage{graphicx}   % Enhanced images support
\RequirePackage{ifluatex}   % LuaTeX-specific options
\RequirePackage{ifpdf}      % pdfTeX-specific options
\RequirePackage{ifxetex}    % XeTeX-specific options
\RequirePackage{kantlipsum} % English kantian-style lipsum
\RequirePackage{lipsum}     % Lorem ipsum
\RequirePackage{listings}   % Code listings
\RequirePackage{longtable}  % Multi-page tables
\RequirePackage{multirow}   % Advanced table cells
\RequirePackage{setspace}   % Set space between lines
\RequirePackage{scrextend}  % Allows \addmargin environment
\RequirePackage{tocloft}    % Custom ToC/LoF/LoT
\RequirePackage{url}        % URL-sensitive line breaks
\RequirePackage{xkeyval}    % Keys and class parameters
\RequirePackage{xspace}     % Remove duplicated spaces

%--------------------------------------
% PdfTeX specific configuration
%--------------------------------------
\ifpdf
    \RequirePackage[T1]{fontenc}
    \RequirePackage[utf8]{inputenc}
    \RequirePackage[
        protrusion=true,
        expansion=true
    ]{microtype}
\fi

%--------------------------------------
% XeTeX specific configuration
%--------------------------------------
\ifxetex
    \RequirePackage{fontspec-xetex}
    \RequirePackage[
        protrusion=true,
    ]{microtype}
\fi

%--------------------------------------
% LuaTeX specific configuration
%--------------------------------------
\ifluatex
    \RequirePackage[T1]{fontenc}
    \RequirePackage[utf8]{luainputenc}
    \RequirePackage{luatexbase}
    \RequirePackage{luacode}
    \RequirePackage[
        protrusion=true,
        expansion=true
    ]{microtype}
    % In LuaTeX, we can prevent one-letter words
    % from beging at the end of the line.
    \begin{luacode}
    local debug = false
    local glyph_id = node.id "glyph"
    local glue_id  = node.id "glue"
    local hlist_id = node.id "hlist"

    local prevent_single_letter = function (head)
        while head do
            if head.id == glyph_id then
                if unicode.utf8.match(unicode.utf8.char(head.char),"%a") then     -- is a letter
                    if head.prev.id == glue_id and head.next.id == glue_id then   -- is one letter word

                        local p = node.new("penalty")
                        p.penalty = 10000

                        if debug then
                            local w = node.new("whatsit","pdf_literal")
                            w.data = "q 1 0 1 RG 1 0 1 rg 0 0 m 0 5 l 2 5 l 2 0 l b Q"
                            node.insert_after(head,head,w)
                            node.insert_after(head,w,p)
                        else
                            node.insert_after(head,head,p)
                        end
                    end
                end
            end
            head = head.next
        end
        return true
    end

    luatexbase.add_to_callback("pre_linebreak_filter",prevent_single_letter,"~")

    \end{luacode}
\fi

%--------------------------------
% Produce clickable hyperlinks
% in ToC, LoF and LoT
%--------------------------------
\RequirePackage{hyperref}
\hypersetup{
    colorlinks,
    citecolor=black,
    filecolor=black,
    linkcolor=black,
    urlcolor=black
}

%--------------------------------------
% Default margin is 2.5 cm
%--------------------------------------
\RequirePackage[
    a4paper,
    left=\eitileft,
    right=\eitiright,
    top=\eititop,
    bottom=\eitibottom,
    bindingoffset=\eitibindingoffset
]{geometry}

%--------------------------------
% Turn off hyphenation
%--------------------------------
\ifeitinohyphenation
    \tolerance=1
    \emergencystretch=\maxdimen
    \hyphenpenalty=10000
    \hbadness=10000
\fi

%--------------------------------
% Initial setup
%--------------------------------
\setstretch{1.2}
\counterwithin{figure}{section}
\counterwithin{table}{section}

%--------------------------------
% Table of Contents setup
%--------------------------------
\setlength{\cftbeforesecskip}{2pt}
\renewcommand{\cftsecfont}{\bf\normalsize}
\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecdotsep}{\cftdotsep}

%--------------------------------
% List of Figures setup
%--------------------------------
\renewcommand*\l@figure{\@dottedtocline{1}{0.5em}{2.25em}}
\newcommand{\listoffigurestoc}{
    \listoffigures
    \addcontentsline{toc}{section}{\listfigurename}
}

%--------------------------------
% List of Tables setup
%--------------------------------
\renewcommand*\l@table{\@dottedtocline{1}{0.5em}{2.25em}}
\newcommand{\listoftablestoc}{
    \listoftables
    \addcontentsline{toc}{section}{\listtablename}
}

%--------------------------------
% Header & footer
%--------------------------------
\SetSectionFormatting{section}{0.5cm}{\FormatHangHeading{\Large}}{0.5cm}
\let\oldsection\section
\renewcommand{\section}{
	\thispagestyle{plain}
	\oldsection
}

\fancypagestyle{plain}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\fancyfoot[LE,RO]{\thepage}
}

\fancypagestyle{headings}{
	\fancyhead{}
	\renewcommand{\headrulewidth}{1pt}
	\fancyheadoffset{0cm}
	\fancyhead[RO]{\nouppercase{\thesection.\hspace{8pt}\leftmark}}
	\fancyhead[LE]{\nouppercase{\thesection.\hspace{8pt}\leftmark}}
	\fancyfoot{}
	\fancyfoot[LE,RO]{\thepage}
}
\pagestyle{headings}

\renewcommand{\sectionmark}[1]{
	\markboth{#1}{#1}
}

%--------------------------------
% Set default list style
%--------------------------------
\setlist[itemize,1]{topsep=2pt,label=\large$\bullet$, leftmargin=28pt}
\setlist[itemize,2]{topsep=2pt,leftmargin=18pt}
\setlist[itemize,3]{topsep=2pt,leftmargin=18pt}

\setlist[enumerate,1]{topsep=2pt,leftmargin=24pt}
\setlist[enumerate,2]{topsep=2pt,leftmargin=16pt}
\setlist[enumerate,3]{topsep=2pt,leftmargin=16pt}

%--------------------------------
% Wybór języka pracy
%--------------------------------
\newcommand{\langpol}{
    \newcommand{\@lang}{english}
    %\usepackage[polish]{babel}
    \usepackage[english]{babel}

	%\newtheorem{theorem}{Twierdzenie}
	%\newtheorem{lemma}{Lemat}
	%\newtheorem{corollary}{Wniosek}
	%\newtheorem{definition}{Definicja}
	%\newtheorem{axiom}{Aksjomat}
	%\newtheorem{assumption}{Założenie}

	\AtBeginDocument{
		\renewcommand{\listfigurename}{Spis rysunków}
		\renewcommand{\listtablename}{Spis tabel}
		\renewcommand{\tablename}{Tabela}
	}
}
\newcommand{\langeng}{
    \newcommand{\@lang}{english}
    \usepackage[english]{babel}

	\newtheorem{theorem}{Theorem}
	\newtheorem{lemma}{Lemma}
	\newtheorem{corollary}{Corollary}
	\newtheorem{definition}{Definition}
	\newtheorem{axiom}{Axiom}
	\newtheorem{assumption}{Assumption}

	\AtBeginDocument{
		\renewcommand{\listfigurename}{List of Figures}
		\renewcommand{\listtablename}{List of Tables}
		\renewcommand{\tablename}{Table}
	}
}

%--------------------------------
% Streszczenie po polsku
%--------------------------------
\newcommand{\streszczenie}{
    \thispagestyle{plain}
    \begin{center}\textbf{\large\@title}\end{center}
    \textbf{\\ Streszczenie.\xspace}
}
\newcommand{\slowakluczowe}{\vspace{0.5cm}\par\noindent \textbf{Słowa kluczowe: \xspace}}

%--------------------------------
% Streszczenie po angielsku
%--------------------------------
\renewcommand{\abstract}{
    \thispagestyle{plain}
    \begin{center}\textbf{\large\@engtitle}\end{center}
    \textbf{\\ Abstract.\xspace}
}
\newcommand{\keywords}{\vspace{0.5cm}\par\noindent \textbf{Keywords: \xspace}}

%--------------------------------
% Oświadczenie o autorstwie
%--------------------------------
\newcommand{\makeauthorship}{
    \ifnum \pdf@strcmp{\@lang}{polish} = 0
        \thispagestyle{plain}
        \begin{figure}[ht]
            \vspace{-55pt}
            \noindent\makebox[\textwidth]{
            \includegraphics[width=1.19\textwidth]{szablon/eiti/oswiadczenie_autora_pracy.pdf}
            }
        \end{figure}
    \fi
    \ifnum \pdf@strcmp{\@lang}{english} = 0
        \thispagestyle{plain}
        \begin{figure}[ht]
            \vspace{-55pt}
            \noindent\makebox[\textwidth]{
            \includegraphics[width=1.19\textwidth]{szablon/eiti/thesis_author_declaration-1.pdf}
            }
        \end{figure}
        \newpage
        \thispagestyle{plain}
        \begin{figure}[h]
            \vspace{-55pt}
            \noindent\makebox[\textwidth]{
            \includegraphics[width=1.19\textwidth]{szablon/eiti/thesis_author_declaration-2.pdf}
            }
        \end{figure}
    \fi
}

%--------------------------------
% Strona tytułowa - makra
%--------------------------------
\newcommand{\EngineerThesis}{
    \newcommand{\ThesisType}{Engineer}
}
\newcommand{\MasterThesis}{
    \newcommand{\ThesisType}{Master}
}
\newcommand{\DiplomaThesis}{
    \newcommand{\ThesisType}{Diploma}
}
\newcommand{\instytut}[1]{
    \newcommand{\@instytut}{#1}
}
\newcommand{\kierunek}[1]{
    \newcommand{\@kierunek}{#1}
}
\newcommand{\specjalnosc}[1]{
    \newcommand{\@specjalnosc}{#1}
}
\newcommand{\album}[1]{
    \newcommand{\@album}{#1}
}
\newcommand{\promotor}[1]{
    \newcommand{\@promotor}{#1}
}
\newcommand{\engtitle}[1]{
    \newcommand{\@engtitle}{#1}
}

%--------------------------------
% Strona tytułowa
%--------------------------------
\let\oldmaketitle\maketitle
\renewcommand{\maketitle}{
    \linespread{1.15}

    \thispagestyle{empty}
    \pagenumbering{gobble}

    \begin{center}
        %\sffamily
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            \includegraphics[width=\textwidth]{szablon/eiti/header.png} \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            \includegraphics[width=\textwidth]{szablon/eiti/header-eng.png} \\
        \fi
        \hfill \break
        \hfill \break
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            Instytut \@instytut \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            Institute of \@instytut \\
        \fi
        \hfill \break
        \hfill \break
        \hfill \break
        \ifnum \pdf@strcmp{\ThesisType}{Diploma} = 0
            \ifnum \pdf@strcmp{\@lang}{polish} = 0
                \includegraphics[width=0.5\textwidth]{szablon/eiti/title-praca-koncowa.png} \\
            \fi
            \ifnum \pdf@strcmp{\@lang}{english} = 0
                \includegraphics[width=0.4\textwidth]{szablon/eiti/title-final-project.png} \\
            \fi
        \fi
        \hfill \break
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            \@kierunek \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            \@kierunek \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            \@specjalnosc \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            \@specjalnosc \\
        \fi
        \hfill \break
        \hfill \break
        \large
        \@title \\
        \hfill \break
        \hfill \break
        \LARGE
        \@author \\
        \normalsize
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            Numer~albumu \@album \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            student record book number \@album \\
        \fi
        \hfill \break
        \hfill \break
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            promotor \\
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            thesis supervisor \\
        \fi
        \@promotor \\
        \vfill
        \ifnum \pdf@strcmp{\@lang}{polish} = 0
            WARSZAWA \@date
        \fi
        \ifnum \pdf@strcmp{\@lang}{english} = 0
            WARSAW \@date
        \fi
    \end{center}

    \pagenumbering{arabic}
    \setcounter{page}{1}
}

%--------------------------------
% Pusta strona - makro
%--------------------------------
\newcommand{\blankpage}{
    \newpage
    \thispagestyle{empty}
    \
    \newpage
}

%--------------------------------
% Skrót (akronim) - makro
%--------------------------------
\newcommand{\acronymlist}{
    \ifnum \pdf@strcmp{\@lang}{polish} = 0
        \section*{Wykaz symboli i skrótów}
    \fi
    \ifnum \pdf@strcmp{\@lang}{english} = 0
        \section*{List of Symbols and Abbreviations}
    \fi
}
\newcommand{\acronym}[2]{
    \par\noindent\hspace{0.4em}
    {\textbf{#1} -- #2}
}

%--------------------------------
% Spis załączników - makra
%--------------------------------
\newcommand{\nocontentsline}[3]{}
\newcommand{\tocless}[2]{%
    \bgroup\let\addcontentsline=\nocontentsline#1{#2}\egroup
}

\newcommand{\@appendixtitle}{
    \ifnum \pdf@strcmp{\@lang}{polish} = 0
        Załącznik
    \fi
    \ifnum \pdf@strcmp{\@lang}{english} = 0
        Appendix
    \fi
}

\let\standardappendix\appendix
\renewcommand\appendix[1]{%
    \tocless\section{\hspace*{1.5em} #1}
    \addcontentsline{app}{subsection}{\hspace*{-1.1em}\arabic{section}.\hspace*{0.5em} #1}
}

\newcommand\listofappendicestoc{
    \ifnum \pdf@strcmp{\@lang}{polish} = 0
        \section*{Spis załączników}\@starttoc{app}
    \fi
    \ifnum \pdf@strcmp{\@lang}{english} = 0
        \section*{List of Appendices}\@starttoc{app}
    \fi

    \standardappendix
    \renewcommand{\thesection}{\@appendixtitle~\arabic{section}}
}

%--------------------------------
% Moje makra i importy // AK
%--------------------------------
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
% \usepackage{pgfplots}
% \usepackage{filecontents}
% \pgfplotsset{compat=1.16}

\usepackage{pgfplots}       % <-- required in preamble
\pgfplotsset{compat=newest} % <-- optional in preamble
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows}
\tikzstyle{startstop_off} = [rectangle, rounded corners, minimum width=3cm, minimum height=2cm,text centered, draw=black, fill=blue!30]
% \tikzstyle{online} = [rectangle, rounded corners, minimum width=3cm, minimum height=2cm,text centered, draw=black, fill=red!30]

% \tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=2cm,text centered, draw=black, fill=red!30]
\tikzstyle{io_on} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=red!30]
\tikzstyle{process_on} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=red!30]
\tikzstyle{io_off} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]
\tikzstyle{process_off} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]

% \tikzstyle{decision} = [diamond, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=green!30]
\tikzstyle{arrow} = [thick,->,>=stealth]

\pgfplotsset{/pgf/number format/use comma} % https://tex.stackexchange.com/questions/247169/pgf-plot-y-axis-shows-comma-instead-of-decimal-point

\usepackage{tabularx}
\newcommand\setrow[1]{\gdef\rowmac{#1}#1\ignorespaces}
\newcommand\clearrow{\global\let\rowmac\relax}
\clearrow