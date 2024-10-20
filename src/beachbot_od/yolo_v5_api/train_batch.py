from . import Train, parse_args


def train_batch(args):
    for img_width in reversed(train.eval_img_widths):
        train.change_img_width(img_width)
        train.train()


if __name__ == "__main__":
    args = parse_args()
    train = Train(args)
