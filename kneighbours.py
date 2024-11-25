import numpy as np


class KnnBruteClassifier(object):
    '''Классификатор реализует взвешенное голосование по ближайшим соседям.
    Поиск ближайшего соседа осуществляется полным перебором.
    Параметры
    ----------
    n_neighbors : int, optional
        Число ближайших соседей, учитывающихся в голосовании
    weights : str, optional (default = 'uniform')
        веса, используемые в голосовании. Возможные значения:
        - 'uniform' : все веса равны.
        - 'distance' : веса обратно пропорциональны расстоянию до классифицируемого объекта
        -  функция, которая получает на вход массив расстояний и возвращает массив весов
    metric: функция подсчета расстояния (по умолчанию l2).
    '''

    def __init__(self, n_neighbors=1, weights='uniform', metric="l2"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.training_set = None
        self.answer = None

    def fit(self, x, y):
        '''Обучение модели.
        Парметры
        ----------
        x : двумерным массив признаков размера n_queries x n_features
        y : массив/список правильных меток размера n_queries
        Выход
        -------
        Метод возвращает обученную модель
        '''
        self.training_set = np.array(x)
        self.answer = np.array(y)
        return self

    def predict(self, x):
        """ Предсказание класса для входных объектов
        Параметры
        ----------
        X : двумерным массив признаков размера n_queries x n_features
        Выход
        -------
        y : Массив размера n_queries
        """
        neigh_dist, neigh_indarray = self.kneighbors(x, self.n_neighbors)
        answers = []
        if self.weights == 'uniform':
            for obj in neigh_indarray:
                answers_for_this = []
                for index in obj:
                    answers_for_this.append(self.answer[index])
                most_common_answer = max(answers_for_this, key=answers_for_this.count)
                answers.append(most_common_answer)

        if self.weights == 'distance':
            for index_this, obj in enumerate(neigh_indarray):
                answers_for_this = []
                answers_dist = []
                for index_neighbor in obj:
                    answers_for_this.append(self.answer[index_neighbor])
                for dist in neigh_dist[index_this]:
                    answers_dist.append(1/dist)
                unique_answers = np.unique(answers_for_this)
                weight_this = {answer: 0 for answer in unique_answers}
                for ind, answer in enumerate(answers_for_this):
                    for ind_unq, answer_unq in enumerate(unique_answers):
                        if answer == answer_unq:
                            weight_this[ind_unq] += answers_dist[ind]
                most_weight_answer = max(weight_this, key=weight_this.get)
                answers.append(most_weight_answer)

        return answers

    def predict_proba(self, x):
        """Возвращает вероятности классов для входных объектов.
        Параметры
        ----------
        X : двумерный массив признаков размера n_queries x n_features

        Выход
        -------
        p : массив размера n_queries x n_classes, вероятности принадлежности
            каждого объекта к каждому классу
        """
        neigh_dist, neigh_indarray = self.kneighbors(x, self.n_neighbors)
        probas = []
        unique_classes = np.unique(self.answer)

        for i, obj in enumerate(neigh_indarray):
            class_weights = {cls: 0 for cls in unique_classes}

            if self.weights == 'uniform':
                for idx in obj:
                    class_weights[self.answer[idx]] += 1

            elif self.weights == 'distance':
                for j, idx in enumerate(obj):
                    weight = 1 / neigh_dist[i][j] if neigh_dist[i][j] != 0 else 0
                    class_weights[self.answer[idx]] += weight

            # Нормализуем веса, чтобы получить вероятности
            total_weight = sum(class_weights.values())
            class_probs = [class_weights[cls] / total_weight for cls in unique_classes]
            probas.append(class_probs)

        return np.array(probas)

    def kneighbors(self, x, n_neighbors):
        """Возвращает n_neighbors ближайших соседей для всех входных объектов и расстояния до них
        Параметры
        ----------
        X : двумерным массив признаков размера n_queries x n_features
        Выход
        -------
        neigh_dist массив размера n_queries х n_neighbors
        расстояния до ближайших элементов
        neigh_indarray, массив размера n_queries x n_neighbors
        индексы ближайших элементов
        """
        neigh_dist = []
        neigh_indarray = []
        for obj_for_predict in x:
            n_neighbors_arr = self.training_set[:n_neighbors]
            distance_to_neighbors = []
            obj_index_neigh = [i for i in range(n_neighbors)]

            for neighbor in n_neighbors_arr:
                distance_vec_init = obj_for_predict - neighbor
                distance_init = np.sqrt(np.dot(distance_vec_init, distance_vec_init))
                distance_to_neighbors.append(distance_init)

            for ind, obj_in_dataset in enumerate(self.training_set):
                distance_vec = obj_for_predict - obj_in_dataset
                distance = np.sqrt(np.dot(distance_vec, distance_vec))
                flag = 0

                for index, _ in enumerate(n_neighbors_arr):
                    if distance < distance_to_neighbors[index] and flag == 0:
                        distance_to_neighbors[index] = distance
                        n_neighbors_arr[index] = obj_in_dataset
                        obj_index_neigh[index] = ind
                        flag = 1

            neigh_dist.append(distance_to_neighbors)
            neigh_indarray.append(obj_index_neigh)

        return neigh_dist, neigh_indarray


def compare():
    from sklearn.datasets import make_classification
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Создание данных
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3,n_informative=3,n_clusters_per_class=2, random_state=42)

    # Обучение вашей реализации
    my_knn = KnnBruteClassifier(n_neighbors=5, weights='uniform')
    my_knn.fit(X, y)

    # Обучение KNeighborsClassifier из sklearn
    sklearn_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    sklearn_knn.fit(X, y)

    # Прогнозирование вероятностей
    my_probas = my_knn.predict_proba(X)
    sklearn_probas = sklearn_knn.predict_proba(X)

    # Тестирование точности вероятностей
    print("Первые 5 прогнозов вероятностей (моя реализация):")
    print(my_probas[:5])

    print("Первые 5 прогнозов вероятностей (sklearn):")
    print(sklearn_probas[:5])

    # Проверка, что вероятности похожи
    similarity = np.allclose(my_probas, sklearn_probas, atol=0.1)
    print(f"Похожи ли вероятности? {'Да' if similarity else 'Нет'}")

    my_preds = np.argmax(my_probas, axis=1)
    sklearn_preds = np.argmax(sklearn_probas, axis=1)

    print("Точность по argmax вероятностей (моя модель):", accuracy_score(y, my_preds))
    print("Точность по argmax вероятностей (sklearn):", accuracy_score(y, sklearn_preds))


compare()

