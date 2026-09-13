"""
Multilingual sarcasm detection pipeline (English, Hindi, Bangla, Urdu, Arabic).

Fixes applied vs. the original script:
  1. Stopword-dict keys now match the ACTUAL values found in the `language`
     column of the datasets ("english", "hindi", "bangla", "urdu", "arabic"),
     not language codes ("en", "hi", ...) or a mismatched "Ar". Previously
     every lookup missed and NO stopwords were ever removed for ANY language.
  2. The Urdu/Arabic stopword lists had literal "{dir=\"rtl\"}" markup baked
     into each string (an artifact of how the text was pasted into Word).
     These have been cleaned to plain tokens so they actually match text.
  3. Added a de-duplication step per dataset before computing min_samples --
     the Hindi file was 93% exact duplicate rows, which was badly skewing
     the "balance to smallest dataset" logic.
  4. `google.colab` import / `files.download()` is now optional -- it only
     runs when the script is actually executing inside Colab, so it no
     longer crashes when run locally or on a server.
  5. Minor cleanup: removed redundant double NLTK downloads and a redundant
     isinstance check.
"""

import os
import re
import sys
import unicodedata

import nltk
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ----------------------------------------------------------------------
# NLTK setup
# ----------------------------------------------------------------------
NLTK_DATA_PATH = "./nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

for resource in ("punkt_tab", "punkt", "stopwords"):
    try:
        nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
    except Exception as e:
        print(f"Warning: could not download NLTK resource '{resource}': {e}")

# ----------------------------------------------------------------------
# Stopwords, keyed EXACTLY as the lowercased `language` column values
# that appear in the CSVs: "english", "hindi", "bangla", "urdu", "arabic".
# ----------------------------------------------------------------------
STOPWORDS_DICT = {
    "english": None,  # filled at runtime from nltk.corpus.stopwords
    "hindi": {
        "अंदर", "अत", "अदि", "अप", "अपना", "अपनि", "अपनी", "अपने",
        "अभि", "अभी", "आदि", "आप", "इंहिं", "इंहें", "इंहों", "इतयादि",
        "इत्यादि", "इन", "इनका", "इन्हीं", "इन्हें", "इन्हों", "इस", "इसका",
        "इसकि", "इसकी", "इसके", "इसमें", "इसि", "इसी", "इसे", "उंहिं",
        "उंहें", "उंहों", "उन", "उनका", "उनकि", "उनकी", "उनके", "उनको",
        "उन्हीं", "उन्हें", "उन्हों", "उस", "उसके", "उसि", "उसी", "उसे",
        "एक", "एवं", "एस", "एसे", "ऐसे", "ओर", "और", "कइ",
        "कई", "कर", "करता", "करते", "करना", "करने", "करें", "कहते",
        "कहा", "का", "काफि", "काफ़ी", "कि", "किंहें", "किंहों", "कितना",
        "किन्हें", "किन्हों", "किया", "किर", "किस", "किसि", "किसी", "किसे",
        "की", "कुछ", "कुल", "के", "को", "कोइ", "कोई", "कोन",
        "कोनसा", "कौन", "कौनसा", "गया", "घर", "जब", "जहाँ", "जहां",
        "जा", "जिंहें", "जिंहों", "जितना", "जिधर", "जिन", "जिन्हें", "जिन्हों",
        "जिस", "जिसे", "जीधर", "जेसा", "जेसे", "जैसा", "जैसे", "जो",
        "तक", "तब", "तरह", "तिंहें", "तिंहों", "तिन", "तिन्हें", "तिन्हों",
        "तिस", "तिसे", "तो", "था", "थि", "थी", "थे", "दबारा",
        "दवारा", "दिया", "दुसरा", "दुसरे", "दूसरे", "दो", "द्वारा", "न",
        "नहिं", "नहीं", "ना", "निचे", "निहायत", "नीचे", "ने", "पर",
        "पहले", "पुरा", "पूरा", "पे", "फिर", "बनि", "बनी", "बहि",
        "बही", "बहुत", "बाद", "बाला", "बिलकुल", "भि", "भितर", "भी",
        "भीतर", "मगर", "मानो", "मे", "में", "यदि", "यह", "यहाँ",
        "यहां", "यहि", "यही", "या", "यिह", "ये", "रखें", "रवासा",
        "रहा", "रहे", "ऱ्वासा", "लिए", "लिये", "लेकिन", "व", "वगेरह",
        "वरग", "वर्ग", "वह", "वहाँ", "वहां", "वहिं", "वहीं", "वाले",
        "वुह", "वे", "वग़ैरह", "संग", "सकता", "सकते", "सबसे", "सभि",
        "सभी", "साथ", "साबुत", "साभ", "सारा", "से", "सो", "हि",
        "ही", "हुअ", "हुआ", "हुइ", "हुई", "हुए", "हे", "हें",
        "है", "हैं", "हो", "होता", "होति", "होती", "होते", "होना",
        "होने",
    },
    "bangla": {
        "অতএব", "অথচ", "অথবা", "অনুযায়ী", "অনেক", "অনেকে", "অনেকেই", "অন্তত",
        "অন্য", "অবধি", "অবশ্য", "অর্থাত", "আই", "আগামী", "আগে", "আগেই",
        "আছে", "আজ", "আদ্যভাগে", "আপনার", "আপনি", "আবার", "আমরা", "আমাকে",
        "আমাদের", "আমার", "আমি", "আর", "আরও", "ই", "ইত্যাদি", "ইহা",
        "উচিত", "উত্তর", "উনি", "উপর", "উপরে", "এ", "এঁদের", "এঁরা",
        "এই", "একই", "একটি", "একবার", "একে", "এক্", "এখন", "এখনও",
        "এখানে", "এখানেই", "এটা", "এটাই", "এটি", "এত", "এতটাই", "এতে",
        "এদের", "এব", "এবং", "এবার", "এমন", "এমনকী", "এমনি", "এর",
        "এরা", "এল", "এস", "ঐ", "ও", "ওঁদের", "ওঁর", "ওঁরা",
        "ওই", "ওকে", "ওখানে", "ওদের", "ওর", "ওরা", "কখনও", "কত",
        "কবে", "কমনে", "কয়েক", "কয়েকটি", "করছে", "করছেন", "করতে", "করবে",
        "করবেন", "করলে", "করলেন", "করা", "করাই", "করায়", "করার", "করি",
        "করিতে", "করিয়া", "করিয়ে", "করে", "করেই", "করেছিলেন", "করেছে", "করেছেন",
        "করেন", "কাউকে", "কাছ", "কাছে", "কাজ", "কাজে", "কারও", "কারণ",
        "কি", "কিংবা", "কিছু", "কিছুই", "কিন্তু", "কী", "কে", "কেউ",
        "কেউই", "কেখা", "কেন", "কোটি", "কোন", "কোনও", "কোনো", "ক্ষেত্রে",
        "কয়েক", "খুব", "গিয়ে", "গিয়েছে", "গিয়ে", "গুলি", "গেছে", "গেল",
        "গেলে", "গোটা", "চলে", "চান", "চায়", "চার", "চালু", "চেয়ে",
        "চেষ্টা", "ছাড়া", "ছাড়াও", "ছিল", "ছিলেন", "জন", "জনকে", "জনের",
        "জন্য", "জন্যওজে", "জানতে", "জানা", "জানানো", "জানায়", "জানিয়ে", "জানিয়েছে",
        "জে", "জ্নজন", "টি", "ঠিক", "তখন", "তত", "তথা", "তবু",
        "তবে", "তা", "তাঁকে", "তাঁদের", "তাঁর", "তাঁরা", "তাঁাহারা", "তাই",
        "তাও", "তাকে", "তাতে", "তাদের", "তার", "তারপর", "তারা", "তারৈ",
        "তাহলে", "তাহা", "তাহাতে", "তাহার", "তিনঐ", "তিনি", "তিনিও", "তুমি",
        "তুলে", "তেমন", "তো", "তোমার", "থাকবে", "থাকবেন", "থাকা", "থাকায়",
        "থাকে", "থাকেন", "থেকে", "থেকেই", "থেকেও", "দিকে", "দিতে", "দিন",
        "দিয়ে", "দিয়েছে", "দিয়েছেন", "দিলেন", "দু", "দুই", "দুটি", "দুটো",
        "দেওয়া", "দেওয়ার", "দেওয়া", "দেখতে", "দেখা", "দেখে", "দেন", "দেয়",
        "দ্বারা", "ধরা", "ধরে", "ধামার", "নতুন", "নয়", "না", "নাই",
        "নাকি", "নাগাদ", "নানা", "নিজে", "নিজেই", "নিজেদের", "নিজের", "নিতে",
        "নিয়ে", "নিয়ে", "নেই", "নেওয়া", "নেওয়ার", "নেওয়া", "নয়", "পক্ষে",
        "পর", "পরে", "পরেই", "পরেও", "পর্যন্ত", "পাওয়া", "পাচ", "পারি",
        "পারে", "পারেন", "পি", "পেয়ে", "পেয়্র্", "প্রতি", "প্রথম", "প্রভৃতি",
        "প্রযন্ত", "প্রাথমিক", "প্রায়", "প্রায়", "ফলে", "ফিরে", "ফের", "বক্তব্য",
        "বদলে", "বন", "বরং", "বলতে", "বলল", "বললেন", "বলা", "বলে",
        "বলেছেন", "বলেন", "বসে", "বহু", "বা", "বাদে", "বার", "বি",
        "বিনা", "বিভিন্ন", "বিশেষ", "বিষয়টি", "বেশ", "বেশি", "ব্যবহার", "ব্যাপারে",
        "ভাবে", "ভাবেই", "মতো", "মতোই", "মধ্যভাগে", "মধ্যে", "মধ্যেই", "মধ্যেও",
        "মনে", "মাত্র", "মাধ্যমে", "মোট", "মোটেই", "যখন", "যত", "যতটা",
        "যথেষ্ট", "যদি", "যদিও", "যা", "যাঁর", "যাঁরা", "যাওয়া", "যাওয়ার",
        "যাওয়া", "যাকে", "যাচ্ছে", "যাতে", "যাদের", "যান", "যাবে", "যায়",
        "যার", "যারা", "যিনি", "যে", "যেখানে", "যেতে", "যেন", "যেমন",
        "র", "রকম", "রয়েছে", "রাখা", "রেখে", "লক্ষ", "শুধু", "শুরু",
        "সঙ্গে", "সঙ্গেও", "সব", "সবার", "সমস্ত", "সম্প্রতি", "সহ", "সহিত",
        "সাধারণ", "সামনে", "সি", "সুতরাং", "সে", "সেই", "সেখান", "সেখানে",
        "সেটা", "সেটাই", "সেটাও", "সেটি", "স্পষ্ট", "স্বয়ং", "হইতে", "হইবে",
        "হইয়া", "হওয়া", "হওয়ায়", "হওয়ার", "হচ্ছে", "হত", "হতে", "হতেই",
        "হন", "হবে", "হবেন", "হয়", "হয়তো", "হয়নি", "হয়ে", "হয়েই",
        "হয়েছিল", "হয়েছে", "হয়েছেন", "হল", "হলে", "হলেই", "হলেও", "হলো",
        "হাজার", "হিসাবে", "হৈলে", "হোক", "হয়",
    },
    "urdu": {
        "آئی", "آئے", "آج", "آخر", "آخرکبر", "آدهی", "آش", "آًب",
        "آٹھ", "آیب", "ئی", "ئے", "اة", "اخبزت", "اختتبم", "ادھر",
        "ارد", "اردگرد", "ارکبى", "اش", "اضتعوبل", "اضتعوبلات", "اضطرذ", "اضکب",
        "اضکی", "اضکے", "اطراف", "اغیب", "افراد", "الگ", "اور", "اوًچب",
        "اوًچبئی", "اوًچی", "اوًچے", "اى", "اً", "اًذر", "اًہیں", "اٹھبًب",
        "اپٌب", "اپٌے", "اچھب", "اچھی", "اچھے", "اکثر", "اکٹھب", "اکٹھی",
        "اکٹھے", "اکیلا", "اکیلی", "اکیلے", "اگرچہ", "اہن", "ایطے", "ایک",
        "ب", "بپطٌذ", "بگسیر", "ت", "تبزٍ", "تت", "تر", "ترتیت",
        "تریي", "تعذاد", "تن", "تو", "توبم", "توہی", "توہیں", "تٌہب",
        "تک", "تھب", "تھوڑا", "تھوڑی", "تھوڑے", "تھی", "تھے", "تیي",
        "ثب", "ثبئیں", "ثبترتیت", "ثبری", "ثبرے", "ثبعث", "ثبلا", "ثبلترتیت",
        "ثبہر", "ثدبئے", "ثرآں", "ثراں", "ثرش", "ثعذ", "ثغیر", "ثلٌذ",
        "ثلٌذوثبلا", "ثلکہ", "ثي", "ثٌب", "ثٌبرہب", "ثٌبرہی", "ثٌبرہے", "ثٌبًب",
        "ثٌذ", "ثٌذکرو", "ثٌذکرًب", "ثٌذی", "ثڑا", "ثڑوں", "ثڑی", "ثڑے",
        "ثھر", "ثھرا", "ثھراہوا", "ثھرپور", "ثھی", "ثہت", "ثہتر", "ثہتری",
        "ثہتریي", "ثیچ", "ج", "خب", "خبرہب", "خبرہی", "خبرہے", "خبهوظ",
        "خبًب", "خبًتب", "خبًتی", "خبًتے", "خبًٌب", "خت", "ختن", "خجکہ",
        "خص", "خططرذ", "خلذی", "خو", "خواى", "خوًہی", "خوکہ", "خٌبة",
        "خگہ", "خگہوں", "خگہیں", "خیطب", "خیطبکہ", "در", "درخبت", "درخہ",
        "درخے", "درزقیقت", "درضت", "دش", "دفعہ", "دلچطپ", "دلچطپی", "دلچطپیبں",
        "دو", "دور", "دوراى", "دوضرا", "دوضروں", "دوضری", "دوضرے", "دوًوں",
        "دکھبئیں", "دکھبتب", "دکھبتی", "دکھبتے", "دکھبو", "دکھبًب", "دکھبیب", "دی",
        "دیب", "دیتب", "دیتی", "دیتے", "دیر", "دیٌب", "دیکھو", "دیکھٌب",
        "دیکھی", "دیکھیں", "دے", "ر", "راضتوں", "راضتہ", "راضتے", "رریعہ",
        "رریعے", "رکي", "رکھ", "رکھب", "رکھتب", "رکھتبہوں", "رکھتی", "رکھتے",
        "رکھی", "رکھے", "رہب", "رہی", "رہے", "ز", "زبصل", "زبضر",
        "زبل", "زبلات", "زبلیہ", "زصوں", "زصہ", "زصے", "زقبئق", "زقیتیں",
        "زقیقت", "زکن", "زکویہ", "زیبدٍ", "صبف", "صسیر", "صفر", "صورت",
        "صورتسبل", "صورتوں", "صورتیں", "ض", "ضبت", "ضبتھ", "ضبدٍ", "ضبرا",
        "ضبرے", "ضبل", "ضبلوں", "ضت", "ضرور", "ضرورت", "ضروری", "ضلطلہ",
        "ضوچ", "ضوچب", "ضوچتب", "ضوچتی", "ضوچتے", "ضوچو", "ضوچٌب", "ضوچی",
        "ضوچیں", "ضکب", "ضکتب", "ضکتی", "ضکتے", "ضکٌب", "ضکی", "ضکے",
        "ضیذھب", "ضیذھی", "ضیذھے", "ضیکٌڈ", "ضے", "طجت", "طرف", "طریق",
        "طریقوں", "طریقہ", "طریقے", "طور", "طورپر", "ظبہر", "ع", "عذد",
        "عظین", "علاقوں", "علاقہ", "علاقے", "علاوٍ", "عووهی", "غبیذ", "غخص",
        "غذ", "غروع", "غروعبت", "غے", "فرد", "فی", "ق", "قجل",
        "قجیلہ", "قطن", "قطہ", "لئے", "لا", "لازهی", "لو", "لوجب",
        "لوجی", "لوجے", "لوسبت", "لوسہ", "لوگ", "لوگوں", "لڑکپي", "لگتب",
        "لگتی", "لگتے", "لگٌب", "لگی", "لگیں", "لگے", "لی", "لیب",
        "لیٌب", "لیں", "لے", "ه", "هتعلق", "هختلف", "هسترم", "هسترهہ",
        "هسطوش", "هسیذ", "هطئلہ", "هطئلے", "هطبئل", "هطتعول", "هطلق", "هعلوم",
        "هػتول", "هلا", "هوکي", "هوکٌبت", "هوکٌہ", "هٌبضت", "هڑا", "هڑًب",
        "هڑے", "هکول", "هگر", "هہرثبى", "هیرا", "هیری", "هیرے", "هیں",
        "و", "وار", "والے", "وخواى", "وٍ", "ٹھیک", "پبئے", "پبش",
        "پبًب", "پبًچ", "پر", "پراًب", "پطٌذ", "پل", "پورا", "پوچھب",
        "پوچھتب", "پوچھتی", "پوچھتے", "پوچھو", "پوچھوں", "پوچھٌب", "پوچھیں", "پچھلا",
        "پھر", "پہلا", "پہلی", "پہلےضی", "پہلےضے", "پہلےضےہی", "پیع", "چبر",
        "چبہب", "چبہٌب", "چبہے", "چلا", "چلو", "چلیں", "چلے", "چکب",
        "چکی", "چکیں", "چکے", "چھوٹب", "چھوٹوں", "چھوٹی", "چھوٹے", "چھہ",
        "چیسیں", "ڈھوًڈا", "ڈھوًڈلیب", "ڈھوًڈو", "ڈھوًڈًب", "ڈھوًڈی", "ڈھوًڈیں", "ک",
        "کبلٌب", "کتہ", "ہ", "ہیں", "یب", "ے",
    },
    "arabic": {
        "،", "ء", "ءَ", "آ", "آب", "آخر", "آذار", "آض",
        "آل", "آمينَ", "آناء", "آنفا", "آه", "آهاً", "آهٍ", "آهِ",
        "أ", "أبدا", "أبريل", "أبو", "أبٌ", "أجل", "أجمع", "أحد",
        "أخبر", "أخذ", "أخو", "أخٌ", "أربع", "أربعاء", "أربعة", "أربعمئة",
        "أربعمائة", "أرى", "أسكن", "أصبح", "أصلا", "أضحى", "أطعم", "أعطى",
        "أعلم", "أغسطس", "أفريل", "أفعل به", "أفٍّ", "أقبل", "أكتوبر", "أكثر",
        "أل", "ألا", "ألف", "ألفى", "أم", "أما", "أمام", "أمامك",
        "أمامكَ", "أمد", "أمس", "أمسى", "أمّا", "أن", "أنا", "أنبأ",
        "أنت", "أنتم", "أنتما", "أنتن", "أنتِ", "أنشأ", "أنفسكم", "أنفسنا",
        "أنفسهم", "أنه", "أنها", "أنًّ", "أنّى", "أهلا", "أو", "أوت",
        "أوشك", "أول", "أولئك", "أولاء", "أولالك", "أوّهْ", "أى", "أي",
        "أيا", "أيار", "أيضا", "أيلول", "أين", "أيّ", "أيّان", "أُفٍّ",
        "إحدى", "إذ", "إذا", "إذاً", "إذما", "إذن", "إزاء", "إضافي",
        "إلا", "إلى", "إلي", "إليكم", "إليكما", "إليكنّ", "إليكَ", "إلَيْكَ",
        "إلّا", "إما", "إمّا", "إن", "إنه", "إنها", "إنَّ", "إى",
        "إياك", "إياكم", "إياكما", "إياكن", "إيانا", "إياه", "إياها", "إياهم",
        "إياهما", "إياهن", "إياي", "إيهٍ", "ا", "ا?ى", "االا", "االتى",
        "ابتدأ", "ابين", "اتخذ", "اثر", "اثنا", "اثنان", "اثني", "اثنين",
        "اجل", "احد", "اخرى", "اخلولق", "اذا", "اربعة", "اربعون", "اربعين",
        "ارتدّ", "استحال", "اصبح", "اضحى", "اطار", "اعادة", "اعلنت", "اف",
        "اكثر", "اكد", "الآن", "الألاء", "الألى", "الا", "الاخيرة", "الان",
        "الاول", "الاولى", "التى", "التي", "الثاني", "الثانية", "الحالي", "الذاتي",
        "الذى", "الذي", "الذين", "السابق", "الف", "اللاتي", "اللتان", "اللتيا",
        "اللتين", "اللذان", "اللذين", "اللواتي", "الماضي", "المقبل", "الوقت", "الى",
        "الي", "اليه", "اليها", "اليوم", "اما", "امام", "امس", "امسى",
        "ان", "انبرى", "انت", "انقلب", "انه", "انها", "او", "اول",
        "اى", "اي", "ايار", "ايام", "ايضا", "ب", "بأن", "بؤسا",
        "بإن", "بئس", "باء", "بات", "باسم", "بان", "بخٍ", "بد",
        "بدلا", "برس", "بسبب", "بسّ", "بشكل", "بضع", "بطآن", "بعد",
        "بعدا", "بعض", "بعيدا", "بغتة", "بل", "بلى", "بن", "به",
        "بها", "بهذا", "بيد", "بين", "بينما", "بَسْ", "بَلْهَ", "ة",
        "ت", "تاء", "تارة", "تاسع", "تانِ", "تانِك", "تبدّل", "تجاه",
        "تحت", "تحوّل", "تخذ", "ترك", "تسع", "تسعة", "تسعمئة", "تسعمائة",
        "تسعون", "تسعين", "تشرين", "تعسا", "تعلَّم", "تفعلان", "تفعلون", "تفعلين",
        "تكون", "تلقاء", "تلك", "تم", "تموز", "تينك", "تَيْنِ", "تِه",
        "تِي", "ث", "ثاء", "ثالث", "ثامن", "ثان", "ثاني", "ثانية",
        "ثلاث", "ثلاثاء", "ثلاثة", "ثلاثمئة", "ثلاثمائة", "ثلاثون", "ثلاثين", "ثم",
        "ثمان", "ثمانمئة", "ثمانون", "ثماني", "ثمانية", "ثمانين", "ثمنمئة", "ثمَّ",
        "ثمّ", "ثمّة", "ج", "جانفي", "جدا", "جعل", "جلل", "جمعة",
        "جميع", "جنيه", "جوان", "جويلية", "جير", "جيم", "ح", "حاء",
        "حادي", "حار", "حاشا", "حاليا", "حاي", "حبذا", "حبيب", "حتى",
        "حجا", "حدَث", "حرى", "حزيران", "حسب", "حقا", "حمدا", "حمو",
        "حمٌ", "حوالى", "حول", "حيث", "حيثما", "حين", "حيَّ", "حَذارِ",
        "خ", "خاء", "خارج", "خاصة", "خال", "خامس", "خبَّر", "خلا",
        "خلافا", "خلال", "خلف", "خمس", "خمسة", "خمسمئة", "خمسمائة", "خمسون",
        "خمسين", "خميس", "د", "دال", "درهم", "درى", "دواليك", "دولار",
        "دون", "دونك", "ديسمبر", "ديك", "دينار", "ذ", "ذا", "ذات",
        "ذاك", "ذال", "ذانك", "ذانِ", "ذلك", "ذهب", "ذو", "ذيت",
        "ذينك", "ذَيْنِ", "ذِه", "ذِي", "ر", "رأى", "راء", "رابع",
        "راح", "رجع", "رزق", "رويدك", "ريال", "ريث", "رُبَّ", "ز",
        "زاي", "زعم", "زود", "زيارة", "س", "ساء", "سابع", "سادس",
        "سبت", "سبتمبر", "سبحان", "سبع", "سبعة", "سبعمئة", "سبعمائة", "سبعون",
        "سبعين", "ست", "ستة", "ستكون", "ستمئة", "ستمائة", "ستون", "ستين",
        "سحقا", "سرا", "سرعان", "سقى", "سمعا", "سنة", "سنتيم", "سنوات",
        "سوف", "سوى", "سين", "ش", "شباط", "شبه", "شتانَ", "شخصا",
        "شرع", "شمال", "شيكل", "شين", "شَتَّانَ", "ص", "صاد", "صار",
        "صباح", "صباحا", "صبر", "صبرا", "صدقا", "صراحة", "صفر", "صهٍ",
        "ـ", "ليس", "هذا", "هو",
    },
}

# English stopwords come straight from NLTK at runtime.
STOPWORDS_DICT["english"] = set(nltk_stopwords.words("english"))


def clean_text(text, lang):
    """
    Clean and preprocess text based on the detected language.
    Handles non-string values gracefully.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove punctuation WITHOUT stripping combining marks (matras/vowel
    # signs). The original `re.sub(r"[^\w\s]", " ", text)` looked safe but
    # Python's `\w` does not count Unicode combining marks (category Mn/Mc)
    # as word characters, so it silently shredded every Devanagari/Bangla
    # word into fragments (e.g. "हमारे" -> "हम र"). This keeps letters,
    # digits, whitespace, and any combining marks attached to them.
    text = "".join(
        ch
        if (ch.isalnum() or ch.isspace() or unicodedata.category(ch) in ("Mn", "Mc", "Me"))
        else " "
        for ch in text
    )
    tokens = word_tokenize(text)
    stop_words = STOPWORDS_DICT.get(lang, set())
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def preprocess_and_balance_datasets(file_paths):
    """
    Combine and preprocess multiple datasets in different languages.
    Balance datasets by taking the smallest number of samples across all
    datasets (after de-duplication).
    """
    balanced_data = []
    min_samples = float("inf")

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Validate required columns
        if not all(col in df.columns for col in ["text", "label", "language"]):
            print(
                f"Skipping {file_path}: Required columns 'text', 'label', "
                f"and 'language' not found."
            )
            continue

        # Ensure 'label' is binary (0 or 1)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
        df = df[df["label"].isin([0, 1])]

        # Ensure 'language' is string and lowercase
        df["language"] = df["language"].astype(str).fillna("unknown").str.lower()

        # Drop exact duplicate rows BEFORE balancing -- otherwise datasets
        # with lots of repeated rows (e.g. Hindi was 93% duplicates) distort
        # the "smallest dataset" size used for balancing.
        before = len(df)
        df = df.drop_duplicates(subset=["text", "label", "language"])
        removed = before - len(df)
        if removed:
            print(f"  Dropped {removed} duplicate rows ({removed / before:.1%}).")

        # Clean text
        df["cleaned_text"] = df.apply(
            lambda row: clean_text(row["text"], row["language"]), axis=1
        )

        # Filter out rows with empty cleaned text
        df = df[df["cleaned_text"] != ""]

        if df.empty:
            print(f"No valid rows found in {file_path} after preprocessing. Skipping.")
            continue

        # Track the smallest dataset size
        min_samples = min(min_samples, len(df))

        # Add to processed data list
        balanced_data.append(df[["cleaned_text", "label", "language"]])

    if not balanced_data:
        print("No valid datasets found. Exiting.")
        return None

    # Balance the datasets by taking the smallest number of samples
    print(f"Balancing datasets to the smallest size: {min_samples} samples per dataset.")
    balanced_datasets = [
        data.sample(n=min_samples, random_state=42) for data in balanced_data
    ]

    # Combine all balanced datasets
    combined_data = pd.concat(balanced_datasets, ignore_index=True)

    # Save the preprocessed dataset
    preprocessed_file_path = "combined_preprocessed_data.csv"
    combined_data.to_csv(preprocessed_file_path, index=False)
    print(f"Combined dataset saved to '{preprocessed_file_path}'.")

    # Only try the Colab download helper if we're actually running in Colab.
    try:
        from google.colab import files  # type: ignore

        files.download(preprocessed_file_path)
    except ImportError:
        pass  # Not running in Colab -- the CSV is already saved locally.

    print(f"Combined balanced dataset size: {combined_data.shape[0]} rows.")
    return combined_data


def train_multilingual_model(data, output_model_path, output_vectorizer_path):
    """
    Train a sarcasm detection model with multilingual data.
    """
    X = data["cleaned_text"]
    y = data["label"]

    print("Splitting dataset into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training the SVM model...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    print("Plotting confusion matrix...")
    import matplotlib

    matplotlib.use("Agg")  # safe for headless environments
    import matplotlib.pyplot as plt

    ConfusionMatrixDisplay.from_estimator(model, X_test_tfidf, y_test, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("Confusion matrix saved to 'confusion_matrix.png'.")

    print("Saving the model and vectorizer...")
    dump(model, output_model_path)
    dump(vectorizer, output_vectorizer_path)
    print(f"Model saved to {output_model_path}")
    print(f"Vectorizer saved to {output_vectorizer_path}")


if __name__ == "__main__":
    # Paths to datasets in different languages -- override via CLI args if given.
    default_paths = [
        "processedenglish_dataset.csv",
        "updated_Bangla.csv",
        "updated_arabic.csv",
        "updated_urdu.csv",
        "updatedhindi_dataset.csv",
    ]
    file_paths = sys.argv[1:] if len(sys.argv) > 1 else default_paths

    multilingual_data = preprocess_and_balance_datasets(file_paths)

    output_model_path = "sarcasm_multilingual_model.joblib"
    output_vectorizer_path = "tfidf_vectorizer_multilingual.joblib"

    if multilingual_data is not None:
        train_multilingual_model(multilingual_data, output_model_path, output_vectorizer_path)
