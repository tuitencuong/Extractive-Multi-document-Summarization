import os
import sys
import math

# Project root
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, ROOT_DIR)

import numpy as np
from utils.preprocessing import Preprocessing
from gensim.models import Word2Vec
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from methods.maximal_marginal_relevance.mmr_summarizer import MMR
from methods.centroid.centroid_summarizer import CentroidBow
import argparse

def mmr_target_count(candidate_count):
    """Keep 70% of the MMR pool, rounded up; an empty pool stays empty."""
    return max(0, math.ceil(0.7 * candidate_count))


def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
    summary = [best_sentence]
    sum_len = 0

    while sentences and sum_len < summary_length - 1:
        MMRval = {}

        for sent in sentences:
            MMRval[sent] = MMR().MMRScore(sent, query, summary, lambta, IDF)

        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)

        if len(sentences) == 0:
            break
        sum_len += 1

    return summary

class Summarizer(object):
    def __init__(self, n_clusters=35, len_summary=20):
        self.n_clusters = n_clusters
        self.len_summary = len_summary

    @staticmethod
    def find_position(index_currents, last_indexs):
        ranking = {}
        for index_current in index_currents:
            for index, item in enumerate(last_indexs):
                if index_current < item:
                    ranking[index_current] = index_current - last_indexs[index - 1]
                    break
            if index_current not in ranking:
                ranking[index_current] = index_current - max(last_indexs)

        sorted_ranking = sorted(ranking.items(), key=operator.itemgetter(1))

        indexs = []
        for item in sorted_ranking:
            indexs.append(item[0])

        return indexs

    def kmean_summarizer(self, sentences, text_sents, org_sents, last_indexs, n_clusters=30, return_trace=False):
        model = Word2Vec(sentences=text_sents,vector_size=256,window=5,min_count=1, workers=4,sg=0)

        X = []
        for sent in text_sents:
            tmp = np.zeros(model.vector_size)

            for word in sent:
                if word in model.wv:
                    tmp += model.wv[word]

            X.append(tmp)

        n_clusters = min(n_clusters,len(text_sents))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(X)

        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])

        index_currents = [closest[idx] for idx in ordering]
        position_index = Summarizer.find_position(index_currents, last_indexs)

        new_sents = [org_sents[idx] for idx in position_index]
        object_sents = [sentences[idx] for idx in position_index]

        if return_trace:
            return new_sents, object_sents, {
                "labels": kmeans.labels_.tolist(),
                "representatives": [int(i) for i in closest],
                "averagePositions": [float(i) for i in avg],
            }
        return new_sents, object_sents

    def mmr_summarizer(self, object_sents, index_sents, mode="train", length_sentence=16, lambta=0.6):
        new_sents_mmr = []

        for item in index_sents:
            new_sents_mmr.append(object_sents[item])

        if not new_sents_mmr:
            return []

        best_sent = new_sents_mmr[0]
        remaining_sents = new_sents_mmr[1:]

        IDF_w = MMR.IDFs(new_sents_mmr)

        # Both modes use the same candidate-based length policy.
        target = mmr_target_count(len(new_sents_mmr))
        summary = makeSummary(remaining_sents, best_sent, best_sent, target, lambta, IDF_w)

        mmr_summaries = []
        for item in summary:
            mmr_summaries.append(item.getOGwords())

        return mmr_summaries

    def add_position(self, new_sents, mmr_summaries, len_sent,index_sents, n_clusters):
        n = len(new_sents)
        if n == 0:
            return []
        selected = []
        for i, sentence in enumerate(new_sents):
            if sentence in mmr_summaries and i not in selected:
                selected.append(i)

        # Chỉ lấy các chỉ số hợp lệ
        valid_index_sents = {i for i in index_sents if isinstance(i, int) and 0 <= i < n }

        target_length = min(len_sent, n)

        # Bổ sung các câu theo thứ tự vị trí
        for i in range(n):
            if len(selected) >= target_length:
                break

            if i not in selected and i not in valid_index_sents:
                selected.append(i)

        # Nếu chưa đủ câu thì bổ sung các chỉ số còn lại
        for i in range(n):
            if len(selected) >= target_length:
                break

            if i not in selected:
                selected.append(i)

        return sorted(selected)


    def centroid_summarizer(self, raw_sentences, new_sents, len_sent=20):
        centroid = CentroidBow()

        centroid_sents = centroid.sumarize(raw_sentences, new_sents, len_sent)

        new_indexs = []
        for item in centroid_sents:
            for index, item2 in enumerate(raw_sentences):
                if item == item2:
                    new_indexs.append(index)

        return new_indexs

    def summary(self, sentences, text_sents, org_sents, last_indexs, len_centroid=20, mode="train"):
        n_clusters = self.n_clusters
        len_sent = self.len_summary

        new_sents, object_sents = self.kmean_summarizer(sentences, text_sents, org_sents, last_indexs, n_clusters)

        raw_sentences = []

        for item in object_sents:
            raw_sentences.append(item.getOGwords())

        index_sents = self.centroid_summarizer(raw_sentences, new_sents, len_centroid)

        mmr_summaries = self.mmr_summarizer(object_sents, index_sents, mode)

        # Only restore source order; do not reintroduce sentences MMR rejected.
        len_sent = min(len_sent, len(mmr_summaries))
        mmr_summaries = mmr_summaries[:len_sent]
        final_index = self.add_position(new_sents, mmr_summaries, len_sent, index_sents, n_clusters)

        final_summaries = [new_sents[i]for i in final_index if isinstance(i, int) and 0 <= i < len(new_sents)]

        return final_summaries

if __name__ == "__main__":
    root_directory = ROOT_DIR + "/"

    document_root = os.path.join(root_directory,"Data","ViMs","original")
    doc_folders = sorted(name for name in os.listdir(document_root)if os.path.isdir(os.path.join(document_root, name))and not name.startswith("."))

    summarizer = Summarizer(n_clusters=10,len_summary=16)

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_to_save',default='GeneratedSummary')

    args = parser.parse_args()

    folder_to_save = args.folder_to_save
    path_to_save = os.path.join(root_directory,"Data","ViMs",folder_to_save)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for folder in doc_folders:
        path = os.path.join(document_root,folder)
        print(path)

        sentences, last_indexs = Preprocessing().openDirectory(path)
        # print("Cluster:", folder)
        # print("Đường dẫn cluster:", path)
        # print("Số tài liệu:", len(last_indexs))
        # print("Số câu:", len(sentences))

        if sentences:
            print("Token câu đầu:", sentences[0].getProcessedWords())
            print("Nội dung câu đầu:", sentences[0].getOGwords())
        else:
            print("KHÔNG ĐỌC ĐƯỢC CÂU NÀO")
            continue
        text_sents = [item.getProcessedWords() for item in sentences]
        org_sents = [item.getOGwords() for item in sentences]

        # Loại câu rỗng
        valid_indices = [
            i for i, words in enumerate(text_sents)
            if words and any(str(w).strip() for w in words)
        ]

        sentences = [sentences[i] for i in valid_indices]
        text_sents = [text_sents[i] for i in valid_indices]
        org_sents = [org_sents[i] for i in valid_indices]

        num_sentences = len(sentences)

        if num_sentences < 5:
            print(f"Bỏ qua {folder}: chỉ có {num_sentences} câu hợp lệ")
            continue

        # Chọn số cụm theo số câu
        if num_sentences < 20:
            n_clusters = 3
        elif num_sentences < 50:
            n_clusters = 5
        elif num_sentences < 100:
            n_clusters = 8
        else:
            n_clusters = 10

        # Không thể có nhiều cụm hơn số câu
        n_clusters = min(n_clusters, num_sentences)

        # Số câu đầu ra không vượt quá số câu đại diện
        len_summary = min(16, n_clusters)

        print(
            f"{folder}: {num_sentences} câu, "
            f"{n_clusters} cụm, "
            f"tối đa {len_summary} câu tóm tắt"
        )

        summarizer = Summarizer(
            n_clusters=n_clusters,
            len_summary=len_summary
        )

        summary = summarizer.summary(
            sentences,
            text_sents,
            org_sents,
            last_indexs
        )

        with open(os.path.join(path_to_save, folder + ".me"),'w',encoding='utf-8') as fileOut:
            fileOut.write("\n".join(summary))
