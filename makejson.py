import json


dict1 = {
"sentences":[
    "the quick brown fox jumped over the lazy dog",
    "the five boxing wizards jump quickly"
]
}

with open('payload_2.json', 'w') as outfile:
    json.dump(dict1, outfile)

dict2 = {
    "sentence_1":"the quick brown fox jumped over the lazy dog",
    "sentence_2":"the five boxing wizards jump quickly"
}

with open('payload_3.json', 'w') as outfile:
    json.dump(dict2, outfile)