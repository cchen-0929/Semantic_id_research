import os


def _parse_line(line):
    arr = line.strip().split()
    if len(arr) < 2:
        return None, []
    user = int(arr[0]) - 1
    items = [int(x) - 1 for x in arr[1:]]
    return user, items


def load_recdata(data_root, dataset):
    train_file = os.path.join(data_root, dataset, f"{dataset}_train.txt")
    test_file = os.path.join(data_root, dataset, f"{dataset}_test.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(train_file)
    if not os.path.exists(test_file):
        raise FileNotFoundError(test_file)

    train_samples = []
    val_samples = []
    test_samples = []
    user_all_items = {}
    all_items = set()

    with open(train_file, "r") as f:
        for line in f:
            user, items = _parse_line(line)
            if user is None or len(items) == 0:
                continue

            all_items.update(items)
            user_all_items[user] = set(items)

            if len(items) >= 3:
                train_hist = items[:-2]
                for i in range(len(train_hist)):
                    train_samples.append((user, train_hist[:i], train_hist[i]))

                val_samples.append((user, train_hist, items[-2]))

                # User-provided rule:
                # test_history = items[:-2] + [items[-1]], gt = items[-1]
                test_hist = train_hist + [items[-1]]
                test_samples.append((user, test_hist, items[-1]))
            else:
                for i in range(len(items)):
                    train_samples.append((user, items[:i], items[i]))

    neg_items = {}
    with open(test_file, "r") as f:
        for line in f:
            user, items = _parse_line(line)
            if user is None:
                continue
            neg_items[user] = items
            all_items.update(items)

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "neg_items": neg_items,
        "user_all_items": user_all_items,
        "all_items": sorted(list(all_items)),
    }

