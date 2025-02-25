# mini_projekt_WDNA_Bartosz_Fijalkowski.py
import random
import csv

class Dataset:
    def __init__(self):
        self.data = []
        self.labels = []
        self.class_column_index = -1

    def read_data(self, filepath: str, header=True, delimiter=',', class_col_index=-1, encoding="utf-8"):
        self.class_column_index = class_col_index
        try:
            with open(filepath, encoding=encoding) as filehandler:
                for line_idx, line in enumerate(filehandler):
                    row = line.strip().split(delimiter)
                    if line_idx == 0 and header:
                        self.labels = row
                    else:
                        self.data.append(row)
        except IOError as err:
            print(f"Błąd odczytu pliku z danymi: {err}")

    def get_labels(self) -> list[str]:
        return self.labels

    def get_number_of_classes(self):
        class_counts = {}
        for row in self.data:
            class_name = row[self.class_column_index]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return [(key, value) for key, value in class_counts.items()]

    def data_split(self, train_pct=0.7, test_pct=0.3, val_pct=0.0):
        total_ratio = train_pct + test_pct + val_pct
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f'Suma wartości procentowych musi wynosić 1.0, otrzymano {total_ratio}')

        random.shuffle(self.data)
        train_last_index = int(len(self.data) * train_pct)
        test_last_index = int(len(self.data) * (train_pct + test_pct))

        train_data = self.data[:train_last_index]
        test_data = self.data[train_last_index:test_last_index]
        valid_data = self.data[test_last_index:]

        return train_data, test_data, valid_data

    def save_to_csv(self, data_list, file_name):
        """Zapisuje podaną listę danych do pliku CSV."""
        try:
            with open(file_name, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if self.labels:
                    writer.writerow(self.labels)
                writer.writerows(data_list)
            print(f"Dane zapisano do pliku: {file_name}")
        except IOError as err:
            print(f"Błąd zapisu do pliku: {err}")

if __name__ == "__main__":
    ds = Dataset()
    ds.read_data('iris.csv')
    print(ds.get_labels())
    print(ds.data)

    print(ds.get_number_of_classes())

    for subset in ds.data_split():
        print(f"Ilość elementów w zbiorze: {len(subset)}")

    for subset in ds.data_split(0.7, 0.2, 0.1):
        print(f"Ilość elementów w zbiorze: {len(subset)}")

    for subset in ds.data_split(0.8, 0.1, 0.1):
        print(f"Ilość elementów w zbiorze: {len(subset)}")

    train_data, _, _ = ds.data_split()
    ds.save_to_csv(train_data, 'train_data.csv')

# Wczytanie modułu
# from mini_projekt_WDNA_Bartosz_Fijalkowski import Dataset
#
# ds = Dataset()
# ds.read_data('iris.csv')  # dane z pliku
# print(ds.get_labels())     # etykiety kolumn
# print(ds.get_number_of_classes())  # liczba klas decyzyjnych
#
# train_data, test_data, val_data = ds.data_split(0.6, 0.3, 0.1)
# print(f"Trening: {len(train_data)}, Test: {len(test_data)}, Walidacja: {len(val_data)}")
# ds.save_to_csv(train_data, 'train_data.csv')