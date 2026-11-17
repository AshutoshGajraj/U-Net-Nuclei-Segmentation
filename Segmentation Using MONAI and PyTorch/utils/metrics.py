import torch

class DiceScore:
    def __init__(self):
        self.scores = []

    def calculate(self, y_true, y_pred):
        """
        Calculate the Dice score for multi-class segmentation,
        excluding the background class (0).
        
        Args:
        y_true (torch.Tensor): Ground truth segmentation mask (B, H, W)
        y_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W)
        
        Returns:
        float: Mean Dice score across all classes (excluding background)
        """
        # Ensure y_pred is in the same format as y_true
        y_pred = torch.argmax(y_pred, dim=1)
        
        # Flatten the tensors and include classes present only in predictions.
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Exclude background while retaining any foreground class found in
        # either the target or the prediction.
        classes = torch.unique(torch.cat((y_true, y_pred)))
        classes = classes[classes != 0]
        
        dice_scores = []
        for cls in classes:
            y_true_cls = (y_true == cls)
            y_pred_cls = (y_pred == cls)
            # --- START OF YOUR CODE ---
            intersection = torch.logical_and(y_true_cls, y_pred_cls).sum()
            dice_score = (2. * intersection) / (y_true_cls.sum() + y_pred_cls.sum() + 1e-6)
            dice_scores.append(dice_score.item())
            # --- END OF YOUR CODE ---
        mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        self.scores.append(mean_dice)
        return mean_dice

    def mean(self):
        """
        Calculate the mean Dice score across all calculations.
        
        Returns:
        float: Mean Dice score
        """
        return torch.mean(torch.tensor(self.scores)).item() if self.scores else 0.0

    def reset(self):
        """
        Reset the stored scores.
        """
        self.scores = []
