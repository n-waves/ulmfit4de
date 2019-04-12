import lxml.etree as etree
import re
import os

EMOJI_LANG = "pl"

def script_relative(filename):
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, filename)

def get_emojis(filenames):
    emojis = {}
    for filename in filenames:
        doc = etree.parse(script_relative(filename))
        for ann in doc.xpath('//annotation[@type="tts"]'):
            emojis[ann.attrib['cp']] = ann.text
    return emojis

def get_emoji_regexps(emojis):
    keys = [re.escape(k) for k in sorted(emojis.keys(), key=len, reverse=True)]
    em = '|'.join(keys)
    return re.compile(f"({em})"), re.compile(f"({em})( ?\\1)+")

emojis = get_emojis([f"{EMOJI_LANG}.xml", f"{EMOJI_LANG}-derived.xml"])
emoji_regexp, repeated_emoji_regexp = get_emoji_regexps(emojis)
