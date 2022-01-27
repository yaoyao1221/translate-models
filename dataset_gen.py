from sentence_transformers import models, SentenceTransformer
import re
import random



class dataset_process():

    def __init__(self):
        self.random_time = 100000
        self.max_sim_tried = 100
        self.max_dif_tried = 100
        self.model_name = 'imxly/sentence_rtb3'
        self.word_embedding_model = models.Transformer(self.model_name)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model])

    def evaluate(self, model,s1,s2):
        '''
        余弦相似度计算
        '''
        import numpy as np
        v1 = model.encode(s1)
        v2 = model.encode(s2)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return v1.dot(v2)

    def data_process(self):
        # 
        words = []
        meaning = {}
        sentence = {}
        #每个句子对应的唯一释义
        sen2mean = {}
        with open('./triplet_prepare.txt') as text:
            f = text.readlines()
            for line in f:
                matched = re.findall(r"'(.*?)'", line)
                # print(matched)
                try:
                    if matched[0] not in words:
                        words.append(matched[0])
                        meaning[matched[0]] = []
                        sentence[matched[0]] = []
                    meaning[matched[0]].append(matched[1])
                    sentence[matched[0]].append(matched[2])
                    sen2mean[matched[2]] = matched[1]
                except Exception as e:
                    pass
                    # print("bad data line")
        result = []
        # 随机选择字
        for random_round in range(self.random_time):
            index = random.randint(0, len(words))
            if sentence[words[index]] == []:
                continue
            result_line = []
            
            #随机选择同字的不同义
            sim_tried = 0
            random_sim_sentence_left = random.choice(sentence[words[index]])
            random_sim_sentence_right = random.choice(sentence[words[index]])

            #避免重复
            while random_sim_sentence_left == random_sim_sentence_right and sim_tried < self.max_sim_tried:
                random_sim_sentence_left = random.choice(sentence[words[index]])
                random_sim_sentence_right = random.choice(sentence[words[index]])
                sim_tried = sim_tried + 1

            while self.evaluate(self.model, sen2mean[random_sim_sentence_left], sen2mean[random_sim_sentence_right]) < 0.5 and sim_tried < self.max_sim_tried:
                if random_sim_sentence_left == random_sim_sentence_right:
                    while random_sim_sentence_left == random_sim_sentence_right and sim_tried < self.max_sim_tried:
                        random_sim_sentence_left = random.choice(sentence[words[index]])
                        random_sim_sentence_right = random.choice(sentence[words[index]])
                        sim_tried = sim_tried + 1
                else:
                    sim_tried = sim_tried + 1
            
            result_line.append([words[index], random_sim_sentence_left])
            result_line.append([words[index], random_sim_sentence_right])

            dif_tried = 0
            dif_word_index = random.randint(0, len(words))
            dif_random_sentence = random.choice(sentence[words[dif_word_index]])
            while self.evaluate(self.model, random_sim_sentence_left, sen2mean[dif_random_sentence]) >= 0.5 and dif_tried < self.max_dif_tried:
                dif_word_index = random.randint(0, len(words))
                dif_random_sentence = random.choice(sentence[words[dif_word_index]])
                dif_tried = dif_tried + 1
            result_line.append([words[dif_word_index], dif_random_sentence])

           
            if sim_tried < self.max_sim_tried and dif_tried < self.max_dif_tried:
                print(result_line, sim_tried, dif_tried, self.evaluate(self.model, sen2mean[random_sim_sentence_left], sen2mean[random_sim_sentence_right]), self.evaluate(self.model, random_sim_sentence_left, sen2mean[dif_random_sentence]))
                result.append(result_line)

if __name__ == "__main__":
    instance = dataset_process()
    instance.data_process()





        