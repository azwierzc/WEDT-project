import matplotlib.pyplot as plt
import numpy as np
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

class Analysis:
    def __init__(self, XHamText, XSpamText):
        self.a = 1
        # self.class_distribution_plotting(XHamText, XSpamText)
        # self.words_cloud(XHamText)
        # self.words_cloud(XSpamText)

    # def class_distribution_plotting(self, XHamText, XSpamText):
    #     indexes = ["Ham", "Spam"]
    #     values = [len(XHamText), len(XSpamText)]
    #
    #     plt.figure()
    #     barList = plt.bar(indexes, values, align="center", width=0.5)
    #     plt.title('Liczba wystąpień danej klasy', fontsize=20)
    #     plt.xlabel('Klasa', fontsize=14)
    #     plt.ylabel('Liczba wystąpień', fontsize=14)
    #     barList[0].set_color('darkorange')
    #     barList[1].set_color('darkblue')
    #     plt.show()
    #
    # def words_cloud(self, data):
    #     words = ''
    #     for msg in data:
    #         msg = msg.lower()
    #         words += msg + ''
    #     wordsCloud = WordCloud(width=600, height=400).generate(words)
    #     plt.imshow(wordsCloud)
    #     plt.axis('off')
    #     plt.show()


    def losses_plotting(self, train_losses_vector, val_losses_vector, steps_vector, title, xlabel):
        plt.figure()
        plt.plot(steps_vector, train_losses_vector, 'r', label='Training loss', )
        plt.plot(steps_vector, val_losses_vector, 'b', label='Validation loss')
        plt.legend()
        plt.xlabel('Steps'.format(xlabel)), plt.ylabel('Loss')
        plt.title(title)
        plt.show()


