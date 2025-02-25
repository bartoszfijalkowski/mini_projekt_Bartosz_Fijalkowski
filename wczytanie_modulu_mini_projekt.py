 # Wczytanie modułu
from mini_projekt_WDNA_Bartosz_Fijalkowski import Dataset

ds = Dataset()
ds.read_data('iris.csv')  # dane z pliku
print(ds.get_labels())     # etykiety kolumn
print(ds.get_number_of_classes())  # liczba klas decyzyjnych

train_data, test_data, val_data = ds.data_split(0.6, 0.3, 0.1)
print(f"Trening: {len(train_data)}, Test: {len(test_data)}, Walidacja: {len(val_data)}")
ds.save_to_csv(train_data, 'train_data.csv')