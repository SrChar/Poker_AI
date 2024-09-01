def get_card_value(card):
    values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
              "J": 11, "Q": 12, "K": 13, "A": 14}
    return values[card.get_number()]

def get_num_value(num):
    values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
              "J": 11, "Q": 12, "K": 13, "A": 14}
    return values[num]
