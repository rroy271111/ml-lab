from src.data_loader import load_dataset
from src.eda import ( print_summary, survival_by_class)

train = load_dataset("train.csv")    
test = load_dataset("test.csv")

print_summary(train)
print(survival_by_class(train))
