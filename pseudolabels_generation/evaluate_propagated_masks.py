from utils.utils import prepare_ground_truths
from utils.eval import get_evaluation_dictionary
from utils.plots import create_boxplot

def evaluate_propagated_masks(propagated_masks_dir, propagation_method, TED_path, ted2camus_path, size=(512,512), overwrite=False):
    
    # Process the TED dataset to get the ground truth masks
    print("Preparing the ground truth masks...")
    prepare_ground_truths(TED_path=TED_path,
                          ted2camus_path=ted2camus_path,
                          size=size,
                          overwrite=overwrite)
    
    # Evaluate all the propagated masks and store the results in a dict
    print("Evaluating the propagated masks...")
    evaluation_dict = get_evaluation_dictionary(propagated_masks_dir=propagated_masks_dir, propagation_method=propagation_method)

    # Use the dictionary to generate and save the plots
    print("Generating the plots...")
    create_boxplot(evaluation_dict, propagated_masks_dir)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--propagated_masks_dir', type=str, default='../data/propagated_masks', help='path to the directory containing the propagated masks')
    parser.add_argument('--propagation_method', type=str, default='sequential', help='propagation method')
    parser.add_argument('--TED_path', type=str, default='../data/ted/database', help='path to the TED database')
    parser.add_argument('--ted2camus_path', type=str, default='../data/ted2camus.csv', help='path to the ted2camus.csv file')
    parser.add_argument('--size', type=tuple, default=(512,512), help='size of the images')
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the existing files')
    args = parser.parse_args()

    evaluate_propagated_masks(
            propagated_masks_dir = args.propagated_masks_dir, 
            propagation_method = args.propagation_method, 
            TED_path = args.TED_path,
            ted2camus_path = args.ted2camus_path, 
            size = args.size, 
            overwrite = args.overwrite,
        )