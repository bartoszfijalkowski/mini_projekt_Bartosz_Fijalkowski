# dataset.py
import random
import csv


class Dataset:
    def __init__(self):
        self.data = []
        self.labels = []
        self.class_column_index = -1

    def read_data(self, filepath: str, header=True, delimiter=',', class_col_index=-1, encoding="utf-8"):
        """
        Wczytuje dane z pliku CSV. Jeśli header = True, pierwszy wiersz to etykiety kolumn.
        Usuwa znak nowej linii \n na końcu każdej linii w celu uniknięcia problemów z wypisywaniem danych.

        :param filepath: Ścieżka do pliku CSV.
        :param header: Flaga określająca, czy pierwszy wiersz zawiera etykiety kolumn (domyślnie True).
        :param delimiter: Separator pól w pliku (domyślnie przecinek).
        :param class_col_index: Indeks kolumny zawierającej klasy decyzyjne (domyślnie ostatnia kolumna).
        :param encoding: Kodowanie pliku (domyślnie UTF-8).
        """
        self.class_column_index = class_col_index
        try:
            with open(filepath, encoding=encoding) as filehandler:
                for line_idx, line in enumerate(filehandler):
                    row = line.rstrip('\n').split(delimiter)
                    if line_idx == 0 and header:
                        self.labels = [label.strip() for label in row]
                    else:
                        self.data.append(row)
        except IOError as err:
            print(f"Błąd odczytu pliku z danymi: {err}")

    def get_labels(self) -> list[str]:
        """
        Zwraca listę etykiet kolumn.

        :return: Lista ciągów znaków zawierająca nazwy kolumn.
        """
        return self.labels

    def get_number_of_classes(self):
        """
        Zlicza unikalne wartości w kolumnie klas decyzyjnych i zwraca je wraz z liczebnością.

        :return: Lista krotek (nazwa klasy, liczba wystąpień).
        """
        class_counts = {}
        for row in self.data:
            class_name = row[self.class_column_index]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return [(key, value) for key, value in class_counts.items()]

    def data_split(self, train_pct=0.7, test_pct=0.2, val_pct=0.1, seed=None):
        """
        Dzieli dane na trzy podzbiory:
        - Dane treningowe: domyślnie 70%)
        - Dane testowe: domyślnie 20%)
        - Dane walidacyjne: domyślnie 10%)
        Suma procentów 1.0.

        :param train_pct: Proporcja danych treningowych.
        :param test_pct: Proporcja danych testowych.
        :param val_pct: Proporcja danych walidacyjnych.
        :param seed: Ziarno losowości dla powtarzalności wyników.
        :return: Trzy listy reprezentujące dane treningowe, testowe i walidacyjne.
        """
        total_ratio = train_pct + test_pct + val_pct
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f'Suma wartości procentowych musi wynosić 1.0, otrzymano {total_ratio}')

        data_copy = self.data.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_copy)

        train_last_index = int(len(data_copy) * train_pct)
        test_last_index = int(len(data_copy) * (train_pct + test_pct))

        train_data = data_copy[:train_last_index]
        test_data = data_copy[train_last_index:test_last_index]
        valid_data = data_copy[test_last_index:]

        return train_data, test_data, valid_data

    def save_to_csv(self, data_list, file_name):
        """
        Zapisuje podaną listę danych do pliku CSV. Etykiety są dodawane jako pierwszy wiersz.

        :param data_list: Lista danych do zapisania.
        :param file_name: Nazwa pliku, do którego zostaną zapisane dane.
        """
        try:
            with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if self.labels:
                    writer.writerow(self.labels)
                writer.writerows(data_list)
            print(f"Dane zapisano do pliku: {file_name}")
        except IOError as err:
            print(f"Błąd zapisu do pliku: {err}")

    def print_data(self, start=0, end=None):
        """
        Wypisuje dane datasetu. Można podać zakres wierszy.

        :param start: Indeks początkowy.
        :param end: Indeks końcowy (domyślnie cały zbiór).
        """
        for row in self.data[start:end]:
            print(row)

    def get_rows_by_class(self, class_value):
        """
        Zwraca wiersze, które należą do podanej klasy decyzyjnej.

        :param class_value: Wartość klasy decyzyjnej.
        :return: Lista wierszy należących do tej klasy.
        """
        return [row for row in self.data if row[self.class_column_index] == class_value]


if __name__ == "__main__":
    ds = Dataset()
    ds.read_data('iris.csv')
    print("Etykiety kolumn:", ds.get_labels())
    print("Klasy decyzyjne:", ds.get_number_of_classes())

    train_set, test_set, val_set = ds.data_split(seed=42)
    print(f"Liczba rekordów treningowych: {len(train_set)}")
    print(f"Liczba rekordów testowych: {len(test_set)}")
    print(f"Liczba rekordów walidacyjnych: {len(val_set)}")

    ds.save_to_csv(train_set, 'train_data.csv')

    print("Przykładowe dane:")
    ds.print_data(0, 6)
    print("Dane dla klasy Iris-setosa:")
    print(ds.get_rows_by_class("Iris-setosa"))
