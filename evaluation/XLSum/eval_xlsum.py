import os
import json
import argparse
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XLSum.")
    parser.add_argument(
        "--langs",
        nargs="+",
        help="List of dataset configurations",
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="/dataset/path",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/model/path",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
    )
    return parser.parse_args()


def write_jsonl(lst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Writing to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in lst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_sentences_from_file(lang_script, path):
    translated_prompt = {
        "arb_Arab": ["وثيقة", "استناداً إلى النص السابق، قم بتقديم ملخص واحد موجز"],
        "amh_Ethi": ["ሰነድ", "በቀደመው ጽሑፍ ላይ በመመስረት አንድ አጭር ማጠቃለያ ያቅርቡ"],
        "ben_Beng": ["দলিল", "পূর্ববর্তী পাঠ্যের উপর ভিত্তি করে, একটি সংক্ষিপ্ত একক সারাংশ প্রদান করুন"],
        "azj_Latn": ["Sənəd", "Əvvəlki mətnə əsasən, qısa bir xülasə təqdim edin"],
        "zho_Hans": ["文档", "根据前面的文字，提供一个简短的单一摘要"],
        "mya_Mymr": ["စာရွက်စာတမ်း", "ယခင်စာသားကို အခြေခံ၍ အကျဉ်းချုပ် တစ်ခုတည်းကို ဖော်ပြပါ-"],
        "zho_Hant": ["文件", "根據前面的文字，提供一個簡短的總結"],
        "fra_Latn": ["Document", "Sur la base du texte précédent, fournissez un bref résumé unique"],
        "eng_Latn": ["Document", "Based on the previous text, provide a brief single summary:"],
        "hau_Latn": ["Takardu", "Dangane da rubutun da ya gabata, samar da taƙaitaccen taƙaitaccen bayani guda ɗaya"],
        "guj_Gujr": ["દસ્તાવેજ", "પાછલા લખાણના આધારે, સંક્ષિપ્ત એક સારાંશ આપો"],
        "ibo_Latn": ["Akwụkwọ", "Dabere na ederede gara aga, wepụta otu nchịkọta nkenke"],
        "hin_Deva": ["दस्तावेज़", "पिछले पाठ के आधार पर, एक संक्षिप्त सारांश प्रदान करें"],
        "ind_Latn": ["Dokumen", "Berdasarkan teks sebelumnya, berikan satu ringkasan singkat"],
        "jpn_Jpan": ["書類", "前のテキストに基づいて、簡単な要約を 1 つ示します"],
        "kir_Cyrl": ["Документ", "Мурунку тексттин негизинде, кыскача бирдиктүү корутунду бериңиз"],
        "kor_Hang": ["문서", "이전 텍스트를 기반으로 간단한 단일 요약을 제공하세요"],
        "run_Latn": ["Inyandiko", "Hashingiwe ku canditswe c’imbere, nutange incamake imwe gusa"],
        "npi_Deva": ["कागजात", "अघिल्लो पाठमा आधारित, संक्षिप्त एकल सारांश प्रदान गर्नुहोस्"],
        "mar_Deva": ["दस्तऐवज", "मागील मजकूरावर आधारित, एक संक्षिप्त सारांश द्या"],
        "pbt_Arab": ["سند", "د مخکیني متن پر بنسټ، یو لنډ لنډیز وړاندې کړئ"],
        "gaz_Latn": ["Sanada", "Barreeffama kanaan duraa irratti hundaa’uun gabaabduu tokko qofa kenni"],
        "pes_Arab": ["سند", "بر اساس متن قبلی، یک خلاصه مختصر ارائه دهید"],
        "pcm_Latn": ["Document", "Based for di previous text, provide a brief single kpatakpata"],
        "pan_Guru": ["ਦਸਤਾਵੇਜ਼", "ਪਿਛਲੇ ਪਾਠ ਦੇ ਆਧਾਰ &#39;ਤੇ, ਇੱਕ ਸੰਖੇਪ ਸੰਖੇਪ ਜਾਣਕਾਰੀ ਪ੍ਰਦਾਨ ਕਰੋ"],
        "por_Latn": ["Documento", "Com base no texto anterior, forneça um breve resumo único"],
        "rus_Cyrl": ["Документ", "На основе предыдущего текста предоставьте краткое резюме"],
        "srp_Latn": ["Dokument", "Na osnovu prethodnog teksta, navedite kratak sažetak"],
        "srp_Cyrl": ["Документ", "На основу претходног текста, дајте кратак појединачни резиме"],
        "gla_Latn": ["Sgrìobhainn", "Stèidhichte air an teacsa roimhe, thoir seachad geàrr-chunntas singilte"],
        "som_Latn": ["Dukumeenti", "Iyada oo ku saleysan qoraalkii hore, bixi mid kooban oo kooban"],
        "sin_Sinh": ["ලේඛනය", "පෙර පාඨය මත පදනම්ව, කෙටි තනි සාරාංශයක් සපයන්න"],
        "tam_Taml": ["ஆவணம்", "முந்தைய உரையின் அடிப்படையில், சுருக்கமான ஒற்றைச் சுருக்கத்தை வழங்கவும்"],
        "swh_Latn": ["Hati", "Kulingana na maandishi yaliyotangulia, toa muhtasari mmoja mfupi"],
        "spa_Latn": ["Documento", "Con base en el texto anterior, proporcione un breve resumen único"],
        "tha_Thai": ["เอกสาร", "จากข้อความก่อนหน้านี้ ขอสรุปสั้น ๆ เพียงข้อเดียว"],
        "tel_Telu": ["పత్రం", "మునుపటి వచనం ఆధారంగా, సంక్షిప్త ఒకే సారాంశాన్ని అందించండి"],
        "tur_Latn": ["Belge", "Önceki metne dayanarak kısa ve tek bir özet verin"],
        "tir_Ethi": ["ሰነድ", "ካብቲ ዝሓለፈ ጽሑፍ ተመርኲስና ሓጺር ንጽል ጽማቕ ኣቕርቡ፤"],
        "ukr_Cyrl": ["документ", "Спираючись на попередній текст, надайте короткий єдиний підсумок"],
        "uzn_Latn": ["Hujjat", "Oldingi matnga asoslanib, qisqacha bitta xulosani keltiring"],
        "urd_Arab": ["دستاویز", "پچھلے متن کی بنیاد پر، ایک مختصر واحد خلاصہ فراہم کریں"],
        "cym_Latn": ["Dogfen", "Yn seiliedig ar y testun blaenorol, rhowch grynodeb sengl byr"],
        "vie_Latn": ["Tài liệu", "Dựa trên văn bản trước, hãy cung cấp một bản tóm tắt ngắn gọn"],
        "yor_Latn": ["Iwe aṣẹ", "Da lori ọrọ iṣaaju, pese akopọ kukuru kan"]
    }
    inputs = []
    targets = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            input = f"{translated_prompt[lang_script][0]}: {data['text']}\n{translated_prompt[lang_script][1]}:"
            inputs.append(input)
            targets.append(data["summary"])
    return inputs, targets


def main():
    args = parse_args()
    print(args)

    if args.model_id == "google/gemma-2-9b" or args.model_id == "google/gemma-2-9b-it":
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    print("Loading model...")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=64,
    )
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    print("Model Loaded...")

    results_dir = os.path.join(args.results_dir, os.path.basename(args.model_id))
    os.makedirs(results_dir, exist_ok=True)

    count = 1
    for line in args.langs:
        lang, lang_script = line.split(" | ")
        print("==============================")
        print(f"Evaluating {count} / 45")
        count += 1

        inputs, targets = read_sentences_from_file(
            lang_script,
            os.path.join(args.dataset_base_path, f"{lang}.jsonl")
        )

        output_path = os.path.join(results_dir, f"{lang_script}.jsonl")
        if os.path.exists(output_path):
            print(f"Skip {lang}")
            continue

        results = []

        outputs = llm.generate(inputs, sampling_params)
        for input, target, output in zip(inputs, targets, outputs):
            output = output.outputs[0].text
            results.append(
                {
                    "model_name": args.model_id,
                    "input": input,
                    "target": target,
                    "output": output,
                }
            )
        write_jsonl(results, output_path)


if __name__ == "__main__":
    print("Starting main...")
    main()


