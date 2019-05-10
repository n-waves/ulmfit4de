import json, html, fire, re, sys
import pandas as pd
from tqdm import tqdm
from emojis import emojis, emoji_regexp, repeated_emoji_regexp
from emoticons import emoticons, emoticons_re
from unidecode import unidecode
#from sacremoses import MosesTokenizer

deleted = ['[deleted]', '[removed]']
link_tag = '<link>'
user_tag = '<user>'
num_tag  = '<number>'


markdown_link = re.compile(r'\[[^]]*\]\([^)]*\)')
link = re.compile(r'(https?|ftp)://[^\s/$.?#].[^\s]*')
reddit_user = re.compile(r"(?<![a-zA-Z0-9_])/?u/[A-Za-z0-9][A-Za-z0-9_]{1,20}(?![a-zA-Z0-9_])")
twitter_user = re.compile(r"(?<![a-zA-Z0-9_])@[A-Za-z0-9_]+(?![a-zA-Z0-9_])")
subreddit_link = re.compile(r"(?<![a-zA-Z0-9_])/?r/[A-Za-z0-9_]+/?(?![a-zA-Z0-9_])")
quotation = re.compile(r'^>.*$', flags=re.MULTILINE)
#multilines = re.compile(r'\n{2,}', flags=re.MULTILINE)
multispaces = re.compile(r'[ \t]{2,}')
reddit_tags = re.compile(r'\\\*hint\\\*')
multichars = re.compile(r'(.)\1{3,}')
multichars_rep = r'\1\1\1'
number_re = re.compile(r'[0-9]+(\.[0-9]+)?')
num_entity_re = re.compile(r'&#(x[0-9a-fA-F]+|[0-9]+);')
remove_alphanum = re.compile(r'\w*')
retweet = re.compile(r'^RT <user> ')
subs_reddit = [(markdown_link, link_tag), (link, link_tag), (reddit_user, user_tag), (subreddit_link, link_tag),
        (reddit_tags, ''), (quotation, ''), (num_entity_re, ' '), (multispaces, ' '), (multichars, multichars_rep), (repeated_emoji_regexp, r'\1')]

subs_twitter = [(link, link_tag), (twitter_user, user_tag), (retweet, ''), (multispaces, ' '), (multichars, multichars_rep), (repeated_emoji_regexp, r'\1')]

def emoji_rep(match):
    name = emojis[match.group(1)]
    return f"<emoji>{name}</emoji>"

def replace_emojis(s):
    return emoji_regexp.sub(emoji_rep, s)

def replace_emoticons(s):
    return emoticons_re.sub(lambda x: emoticons[x.group(1)], s)
#mt = MosesTokenizer('pl')

def preprocess_comment(comment, subs, keep_ogonki, lowercase, keep_numbers, escape_emoji, remove_asciiart, keep_order=False):
    comment = html.unescape(comment.strip())
    if comment in deleted and not keep_order:
        return ''
    if lowercase:
        comment = comment.lower()
    for expr, rep in subs:
        comment = expr.sub(rep, comment)
    comment = replace_emoticons(comment)
    if not keep_numbers:
        comment = number_re.sub(f' {num_tag} ', comment)
    if remove_asciiart and not keep_order and len(remove_alphanum.sub('', comment)) > 0.5 * len(comment):
        return ''
    if escape_emoji:
        comment = replace_emojis(comment)
    if not keep_ogonki:
        comment = unidecode(comment)
    comment = comment.strip()
    return comment

def preprocess(filename, twitter=False, format='json', show_ignored_only=False, keep_ogonki=True, lowercase=False, keep_numbers=True, escape_emoji=True, remove_asciiart=True, keep_order=False, output_format='tsv'):
    with open(filename, 'r') as f:
        if format == 'lines':
            all_comments = f.readlines()
        elif format == 'json':
            bodies = json.load(f)
            all_comments = [comment for body in bodies for comment in body.split('\n')]
        elif format == 'tsv':
            all_comments = [str(x) for x in pd.read_csv(filename, sep='\t', header=None)[0].values]
        elif format == 'csv':
            all_comments = [str(x) for x in pd.read_csv(filename)["body"].values]
        else:
            print(f"Unknown format: {format}", file=sys.stderr)
            exit(-1)

    #all_comments = all_comments[::1000]
    subs = subs_reddit if not twitter else subs_twitter
    comments = [preprocess_comment(comment, subs, keep_ogonki, lowercase, keep_numbers, escape_emoji, remove_asciiart, keep_order) for comment in tqdm(all_comments)]
    empty = 0

    df = pd.DataFrame(comments)
    df2 = pd.DataFrame(all_comments)
    if not show_ignored_only:
        sep = '\t' if output_format == 'tsv' else ','
        if keep_order:
            out = df
        else:
            out = df[df[0] != '']
        out.to_csv(sys.stdout, header=None, index=None, sep=sep)
    else:
        print(df2[df[0] == ''])
        #for orig in df2[df[0] == '']:
        #    print(df2[orig])
    empty = sum(df[0] == '')

##    for orig, comment in tqdm(zip(all_comments, comments)):
##        if comment != '':
##            if not show_ignored_only:
###                print("= Comment =\n")
##                print(comment)
###                print()
##        else:
##            empty += 1
##            if show_ignored_only:
##                print(orig)

    if empty > 0:
        print(f"{empty} comments ignored", file=sys.stderr)

fire.Fire(preprocess)
