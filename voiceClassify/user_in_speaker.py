from voiceClassify.voiceClassify import VoiceClassify
import json

#input : 'kang1', './test/kang1_178.wav' 형식

# from user_in_speaker import user_in_speaker 형식으로 불러오세요

def user_in_speaker(user_id,user_audio_file): # speaker.json파일을 xgboost에서 찾은 결과로 dump
    vc = VoiceClassify(user_audio_file)
    predicted = vc.predict()
    result=predicted[0][0][0]
    with open("./datasets/english/speakers.json", 'w', encoding='utf-8') as make_file:
        speakers=["S015", "S020", "S021", "S023",  "S027",  "S031",  "S032",  "S033",  "S034",  "S035",  "S036",  "S037",  "S038",  "S039",
    "S040",  "S041",  "S042",  "S043",  "S044",   "S045", "S046","S047","S048", "S049","S050", "S051", "S052", "S053", "S054",
    "S055", "S056", "S058",  "S059", "S060", "S061", "S063", "S064", "S065", "S066", "S067", "S069",  "S070", "S071",  "S072",
    "S073",    "S074",    "S075",    "S076",    "S077",    "S078",    "S079",    "S080",    "S082",    "S083",    "S084",    "S085",
    "S086",    "S087",    "S088",    "S090",    "S091",    "S092",    "S093",    "S094",    "S095",    "S096",    "S097",    "S098",
    "S099",    "S100",    "S101",    "S102",    "S103",    "S104",    "S105",    "S106",    "S107",    "S109",    "S110",    "S111",
    "S112",    "S113",    "S114",    "S115",    "S116",    "S117",    "S118",    "S119",    "S120",    "S121",    "S122",    "S123",
    "S125",    "S126",    "S127",    "S128",    "S129",    "S131",    "S132",    "S133",    "V001",    "V002"]
        for n, i in enumerate(speakers):
            if i == result:
                index=speakers.index(result)
                speakers[index] = str(user_id)
        json.dump(speakers, make_file,indent="\t")
#    return speakers

    


