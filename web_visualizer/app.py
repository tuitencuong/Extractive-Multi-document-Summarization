import os
import re
import sys
import math
import unicodedata
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.main_method.Kmeans_CentroidBase_MMR_SentencePosition import Summarizer
from methods.centroid.centroid_summarizer import CentroidBow
from methods.maximal_marginal_relevance.mmr_summarizer import MMR
from utils.sentence import Sentence
from web_visualizer.pipeline_demo import demo_script, demo_styles, demo_module, demo_defaults
from web_visualizer.source_order import sort_by_input

try:
    from underthesea import sent_tokenize, word_tokenize
except Exception:
    sent_tokenize = None
    word_tokenize = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


app = Flask(__name__)

PUNCTUATION = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "(",
    ")",
    "`",
    "_",
    "--",
    "``",
    "''",
    "'s",
}


def split_sentences(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()
    if not text:
        return []

    if sent_tokenize:
        try:
            return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
        except Exception:
            pass

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def tokenize_sentence(sentence):
    sentence = unicodedata.normalize("NFC", sentence).lower()
    if word_tokenize:
        try:
            tokenized = word_tokenize(sentence, format="text")
            tokens = tokenized.split() if isinstance(tokenized, str) else tokenized
        except Exception:
            tokens = re.findall(r"\w+", sentence, flags=re.UNICODE)
    else:
        tokens = re.findall(r"\w+", sentence, flags=re.UNICODE)

    return [
        token
        for token in tokens
        if token.strip() and token not in PUNCTUATION and "&" not in token
    ]


def read_text_file(file_storage):
    data = file_storage.read()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def read_docx_file(file_storage):
    if Document is None:
        raise ValueError("Chưa cài python-docx để đọc file .docx.")
    document = Document(file_storage.stream)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def read_pdf_file(file_storage):
    if PdfReader is None:
        raise ValueError("Chưa cài pypdf để đọc file .pdf.")
    reader = PdfReader(file_storage.stream)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_uploaded_file(file_storage):
    filename = file_storage.filename or "uploaded"
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return read_text_file(file_storage)
    if suffix == ".docx":
        return read_docx_file(file_storage)
    if suffix == ".pdf":
        return read_pdf_file(file_storage)
    raise ValueError(f"Không hỗ trợ định dạng {suffix or '(không có đuôi file)'}." )


def clean_document_text(text):
    text = (text or "").strip()
    match = re.search(r"\bContent\s*:", text, flags=re.IGNORECASE)
    if match:
        return text[match.end() :].strip()
    return text


def build_documents(texts, files):
    documents = []
    for index, text in enumerate(texts, start=1):
        text = clean_document_text(text)
        if text:
            documents.append(
                {
                    "name": f"Văn bản nhập {index}",
                    "source": "manual",
                    "text": text,
                }
            )

    for file_storage in files:
        if not file_storage or not file_storage.filename:
            continue
        text = clean_document_text(read_uploaded_file(file_storage))
        if text:
            documents.append(
                {
                    "name": file_storage.filename,
                    "source": "file",
                    "text": text,
                }
            )

    return documents


def preprocess_documents(documents):
    sentence_objects = []
    sentence_records = []
    last_indexs = []

    for doc_index, document in enumerate(documents):
        last_indexs.append(len(sentence_objects))
        kept = 0

        for input_position, sentence in enumerate(split_sentences(document["text"])):
            tokens = tokenize_sentence(sentence)
            if len(tokens) <= 4:
                continue

            sentence_object = Sentence(document["name"], tokens, sentence)
            sentence_id = f"d{doc_index + 1}s{kept + 1}"

            sentence_objects.append(sentence_object)
            sentence_records.append(
                {
                    "id": sentence_id,
                    "docIndex": doc_index,
                    "docName": document["name"],
                    "position": kept,
                    "inputPosition": input_position,
                    "globalPosition": len(sentence_records),
                    "text": sentence,
                    "tokens": tokens,
                    "tokenCount": len(tokens),
                    "processed": " ".join(tokens),
                }
            )
            kept += 1

    return sentence_objects, sentence_records, last_indexs


def choose_cluster_count(sentence_count):
    if sentence_count < 20:
        return 3
    if sentence_count < 50:
        return 5
    if sentence_count < 100:
        return 8
    return 10


def top_terms_from_existing_mmr(sentence_objects, limit=24):
    terms = []
    weights = MMR.TF_IDF(sentence_objects)
    for score in sorted(weights.keys(), reverse=True):
        for word in weights[score]:
            terms.append({"term": word, "weight": float(score)})
            if len(terms) == limit:
                return terms
    return terms


def centroid_scores(clean_sentences):
    vectorizer = CountVectorizer(stop_words="english")
    sentence_matrix = vectorizer.fit_transform(clean_sentences)
    tfidf = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False).fit_transform(
        sentence_matrix
    ).toarray()
    centroid = tfidf.sum(0)
    maximum = centroid.max()
    if maximum:
        centroid = centroid / maximum
    centroid[centroid <= 0.3] = 0
    return [float(CentroidBow.similarity(row, centroid)) for row in tfidf]


def finite_score(value):
    return float(value) if math.isfinite(value) else 0.0


def mmr_scores(object_sents, index_sents, mmr_summaries, lambta=0.6):
    candidates = [object_sents[index] for index in index_sents]
    if not candidates:
        return {}

    idf = MMR.IDFs(candidates)
    query = candidates[0]
    selected = [candidates[0]]
    scores = {
        id(candidates[0]): {
            "relevance": finite_score(MMR.sentenceSim(candidates[0], query, idf)),
            "redundancy": 0.0,
            "mmrScore": finite_score(MMR.sentenceSim(candidates[0], query, idf)),
        }
    }

    remaining = candidates[1:]
    for text in mmr_summaries[1:]:
        candidate = next((item for item in remaining if item.getOGwords() == text), None)
        if candidate is None:
            continue
        remaining.remove(candidate)
        relevance = MMR.sentenceSim(candidate, query, idf)
        redundancy = max(
            MMR.sentenceSim(candidate, previous, idf)
            for previous in selected
        )
        scores[id(candidate)] = {
            "relevance": finite_score(relevance),
            "redundancy": finite_score(redundancy),
            "mmrScore": finite_score(lambta * relevance - (1 - lambta) * redundancy),
        }
        selected.append(candidate)

    return scores


def index_mmr_summaries(new_sents, mmr_summaries):
    indexes = []
    used = set()
    for summary_sentence in mmr_summaries:
        for index, sentence in enumerate(new_sents):
            if index in used:
                continue
            if sentence == summary_sentence:
                indexes.append(index)
                used.add(index)
                break
    return indexes


def summarize(documents, requested_summary_length=None, lambta=0.6):
    sentence_objects, sentence_records, last_indexs = preprocess_documents(documents)
    if not sentence_objects:
        raise ValueError("Không tìm thấy câu hợp lệ để tóm tắt. Mỗi câu cần có ít nhất 5 từ.")

    text_sents = [sentence.getProcessedWords() for sentence in sentence_objects]
    org_sents = [sentence.getOGwords() for sentence in sentence_objects]
    sentence_count = len(sentence_objects)

    n_clusters = min(choose_cluster_count(sentence_count), sentence_count)
    summary_length = min(requested_summary_length or 16, 16, n_clusters)

    summarizer = Summarizer(n_clusters=n_clusters, len_summary=summary_length)

    new_sents, object_sents, cluster_trace = summarizer.kmean_summarizer(
        sentence_objects,
        text_sents,
        org_sents,
        last_indexs,
        n_clusters,
        return_trace=True,
    )

    object_to_original = {id(sentence): index for index, sentence in enumerate(sentence_objects)}
    candidate_to_original = [object_to_original[id(sentence)] for sentence in object_sents]
    raw_sentences = [sentence.getOGwords() for sentence in object_sents]

    index_sents = summarizer.centroid_summarizer(raw_sentences, new_sents, 20)
    if not index_sents:
        index_sents = list(range(len(object_sents)))

    mmr_summaries = summarizer.mmr_summarizer(
        object_sents,
        index_sents,
        mode="train",
        lambta=lambta,
    )
    summary_length = min(summary_length, len(mmr_summaries))
    mmr_candidate_indexes = index_mmr_summaries(new_sents, mmr_summaries)
    centroid_score_by_index = centroid_scores(new_sents)
    mmr_score_by_object = mmr_scores(object_sents, index_sents, mmr_summaries, lambta)

    final_candidate_indexes = summarizer.add_position(
        new_sents,
        mmr_summaries[:summary_length],
        summary_length,
        index_sents,
        n_clusters,
    )
    final_candidate_indexes = [
        index for index in final_candidate_indexes if 0 <= index < len(candidate_to_original)
    ][:summary_length]
    selected_original_indexes = sorted([
        candidate_to_original[index] for index in final_candidate_indexes
    ])
    selected_order = {
        original_index: order
        for order, original_index in enumerate(selected_original_indexes, start=1)
    }

    for sentence_index, record in enumerate(sentence_records):
        record["cluster"] = cluster_trace["labels"][sentence_index] + 1
        record["selected"] = sentence_index in selected_order
        record["selectedOrder"] = selected_order.get(sentence_index)
        record["mmrScore"] = None

    clusters = []
    remaining_clusters = list(range(n_clusters))
    for candidate_index, original_index in enumerate(candidate_to_original):
        cluster_index = next(i for i in remaining_clusters
                             if cluster_trace["representatives"][i] == original_index)
        remaining_clusters.remove(cluster_index)
        members = [record["id"] for i, record in enumerate(sentence_records)
                   if cluster_trace["labels"][i] == cluster_index]
        clusters.append(
            {
                "cluster": cluster_index + 1,
                "sentenceIndex": original_index,
                "sentenceId": sentence_records[original_index]["id"],
                "size": len(members),
                "score": centroid_score_by_index[candidate_index],
                "avgPosition": cluster_trace["averagePositions"][cluster_index],
                "members": members,
                "candidateIndex": candidate_index,
                "selected": candidate_index in final_candidate_indexes,
                "selectedOrder": selected_order.get(original_index),
                "centroidSelected": candidate_index in index_sents,
                "mmrSelected": candidate_index in mmr_candidate_indexes,
            }
        )

    mmr_steps = []
    for step_order, candidate_index in enumerate(mmr_candidate_indexes, start=1):
        if candidate_index >= len(candidate_to_original):
            continue
        original_index = candidate_to_original[candidate_index]
        score = mmr_score_by_object.get(id(object_sents[candidate_index]), {})
        mmr_steps.append(
            {
                "sentenceIndex": original_index,
                "candidateIndex": candidate_index,
                "order": step_order,
                "relevance": score.get("relevance", 0.0),
                "redundancy": score.get("redundancy", 0.0),
                "mmrScore": score.get("mmrScore", 0.0),
            }
        )

    summary_sentences = []
    for display_index, candidate_index in enumerate(final_candidate_indexes, start=1):
        original_index = candidate_to_original[candidate_index]
        summary_sentences.append(
            {
                "id": sentence_records[original_index]["id"],
                "docName": sentence_records[original_index]["docName"],
                "text": sentence_records[original_index]["text"],
                "order": display_index,
                "displayOrder": display_index,
                "reason": "mmr" if candidate_index in mmr_candidate_indexes else "position",
                "sourceIndex": original_index,
                "docIndex": sentence_records[original_index]["docIndex"],
                "sourcePosition": sentence_records[original_index]["inputPosition"] + 1,
            }
        )

    summary_sentences, sorting = sort_by_input(summary_sentences)
    document_views = []
    for doc_index, document in enumerate(documents):
        document_sentences = [
            sentence for sentence in sentence_records if sentence["docIndex"] == doc_index
        ]
        document_views.append(
            {
                "name": document["name"],
                "source": document["source"],
                "sentenceCount": len(document_sentences),
                "sentences": document_sentences,
            }
        )

    vocabulary = {token for sentence in text_sents for token in sentence}

    return {
        "documents": document_views,
        "stats": {
            "documentCount": len(documents),
            "sentenceCount": sentence_count,
            "summaryLength": len(summary_sentences),
            "clusterCount": n_clusters,
            "candidateCount": len(candidate_to_original),
            "vocabularySize": len(vocabulary),
            "summaryMode": "auto" if requested_summary_length is None else "manual",
            "requestedSummaryLength": requested_summary_length,
            "targetSummaryLength": summary_length,
            "centroidCount": len(index_sents),
            "mmrCount": len(mmr_candidate_indexes),
            "lambda": lambta,
        },
        "steps": {
            "preprocess": {
                "sentences": sentence_records,
                "topTerms": top_terms_from_existing_mmr(sentence_objects),
            },
            "clusters": clusters,
            "mmr": mmr_steps,
            "sorting": sorting,
        },
        "summary": summary_sentences,
        "pipeline": "methods/main_method/Kmeans_CentroidBase_MMR_SentencePosition.py",
    }


@app.route("/")
def index():
    return render_template("index.html", demo_defaults=demo_defaults())


@app.get("/demo/assets.js")
def pipeline_demo_script():
    return Response(demo_script(), mimetype="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/demo/assets.css")
def pipeline_demo_styles():
    return Response(demo_styles(), mimetype="text/css", headers={"Cache-Control": "no-store"})


@app.post("/api/pipeline")
def api_pipeline():
    try:
        documents = build_documents(request.form.getlist("texts"), request.files.getlist("files"))
        if not documents:
            raise ValueError("Hãy nhập văn bản hoặc chọn ít nhất một file.")
        raw_length = request.form.get("summaryLength", "").strip()
        raw_k = request.form.get("clusters", "").strip()
        try:
            length = int(raw_length) if raw_length else None
            clusters = int(raw_k) if raw_k else None
        except ValueError:
            raise ValueError("Số câu và số cụm K phải là số nguyên, hoặc để trống để tự động.")
        if length is not None and not 1 <= length <= 16:
            raise ValueError("Số câu phải từ 1 đến 16, hoặc để trống để tự động.")
        result = demo_module().run_pipeline({
            "documents": [document["text"] for document in documents],
            "clusters": clusters,
            "mode": request.form.get("mode", "train"),
            "lambda": request.form.get("lambda", "0.6"),
            "centroid_mode": "cosine",
            "cosine_threshold": request.form.get("cosineThreshold", "0.35"),
            "summary_length": length,
        })
        result["sourceDocuments"] = documents
        result["summaryMode"] = "auto" if length is None else "manual"
        # Maintain source filenames from the shared uploader for all scenes.
        for sentence in result["sentences"]:
            sentence["doc_label"] = documents[sentence["doc"]]["name"]
        for sentence in result["final"]:
            source = result["sentences"][sentence["source_index"]]
            sentence["doc_label"] = source["doc_label"]
            sentence["sourceIndex"] = sentence["source_index"]
            sentence["docIndex"] = source["doc"]
            sentence["sourcePosition"] = source.get("input_position") or sum(
                item["doc"] == source["doc"]
                for item in result["sentences"][:sentence["source_index"] + 1]
            )
        result["final"], result["sorting"] = sort_by_input(result["final"])
        return jsonify(result)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Lỗi khi chạy demo: {error}"}), 500


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    try:
        texts = request.form.getlist("texts")
        files = request.files.getlist("files")
        raw_length = request.form.get("summaryLength", "").strip()
        try:
            summary_length = int(raw_length) if raw_length else None
        except ValueError:
            raise ValueError("Số câu phải là số nguyên từ 1 đến 16, hoặc để trống để tự động.")
        if summary_length is not None and not 1 <= summary_length <= 16:
            raise ValueError("Số câu phải từ 1 đến 16, hoặc để trống để tự động.")
        documents = build_documents(texts, files)

        if not documents:
            return jsonify({"error": "Hãy nhập văn bản hoặc chọn ít nhất một file."}), 400

        try:
            lambta = float(request.form.get("lambda", "0.6"))
        except ValueError:
            raise ValueError("λ MMR phải là số từ 0 đến 1.")
        if not math.isfinite(lambta) or not 0 <= lambta <= 1:
            raise ValueError("λ MMR phải nằm trong khoảng 0 đến 1.")
        result = summarize(documents, summary_length, lambta)
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Lỗi khi xử lý: {error}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("WEB_VIS_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
