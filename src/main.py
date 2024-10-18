import argparse


def create_parser():
    psr = argparse.ArgumentParser(description="Simple Recommender System CLI")
    sub_psrs = psr.add_subparsers(dest="action", required=True, help="Operation to perform")
    
    train_psr = sub_psrs.add_parser("train", help="Train a model")
    train_psr.add_argument("--algo", "-a", choices=["slopeone", "knn", "svd"], required=True, help="Algorithm to use")
    train_psr.add_argument("--data", "-d", required=True, help="Path lead to dataset")
    train_psr.add_argument("--model", "-m", required=True, help="Name for trained model")
    
    predict_psr = sub_psrs.add_parser("predict", help="Use a model to make prediction")
    predict_psr.add_argument("--model", "-m", required=True, help="Name of model to be used")
    predict_psr.add_argument("--user", "-u", type=int, required=True, help="User ID for prediction")
    predict_psr.add_argument("--item", "-i", type=int, required=True, help="Item ID for prediction")
    
    delete_psr = sub_psrs.add_parser("delete", help="Delete a model")
    delete_psr.add_argument("--model", "-m", required=True, help="Name of model to be deleted")

    config_psr = sub_psrs.add_parser("config", help="Configure algorithm")
    config_psr.add_argument("--algo", "-a", choices=["slopeone", "knn", "svd"], required=True, help="Algorithm to be configured")
    
    args, _ = psr.parse_known_args()
    if args.action == "config":
        if args.algo == "slopeone":
            print("***")
        
        elif args.algo == "knn":
            config_psr.add_argument("--max-k", type=int, default=40, help="Maximum number of filtered neighbors")
            config_psr.add_argument("--min-k", type=int, default=1, help="Minimum number of filtered neighbors")
            config_psr.add_argument("--sim", "-s", choices=["cosine", "msd", "pearson"], default="cosine", help="Similarity to use")
            config_psr.add_argument("--item-based", "-i", action="store_true", help="Choose item-based or user-based (user-based by default)")
            config_psr.add_argument("--min-support", type=int, default=0, help="Minimum number of items or users to consider similarities")

        if args.algo == "svd":
            print("***")
        
    return psr


def perform_training(args):
    print(f"Model {args.model} used {args.algo} on {args.data}")


def perform_prediction(args):
    print(f"Model {args.model} predicted user{args.user} and item{args.item}")
    
    
def perform_deletion(args):
    print(f"Model {args.model} has been deleted")
    
    
def perform_configuration(args):
    if args.algo == "slopeone":
        print("Slope One has no adjustable parameters")
    
    elif args.algo == "knn":
        print(f"{args.algo} has been configured to:")
        print(f"    minimum-k   = {args.min_k}")
        print(f"    maximum-k   = {args.max_k}")
        print(f"    similarity  = {args.sim}")
        print(f"    item-based  = {args.item_based}")
        print(f"    min-support = {args.min_support}")
    
    elif args.algo == "svd":
        print("SVD is still developing")


def main():
    psr = create_parser()
    args = psr.parse_args()
    
    if args.action == "train":
        perform_training(args)
    
    elif args.action == "predict":
        perform_prediction(args)
    
    elif args.action == "delete":
        perform_deletion(args)
    
    elif args.action == "config":
        perform_configuration(args)


if __name__ == "__main__":
    main()