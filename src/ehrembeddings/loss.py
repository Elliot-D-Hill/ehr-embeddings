import torch


class DiscreteFailureTimeNLL(torch.nn.Module):

    def __init__(self, bin_boundaries, tolerance=1e-8):
        super(DiscreteFailureTimeNLL, self).__init__()

        self.bin_starts = torch.tensor(bin_boundaries[:-1])
        self.bin_ends = torch.tensor(bin_boundaries[1:])

        self.bin_lengths = self.bin_ends - self.bin_starts

        self.tolerance = tolerance

    def _discretize_times(self, times):
        return (times[:, None] > self.bin_starts[None, :]) & (
            times[:, None] <= self.bin_ends[None, :]
        )

    def _get_proportion_of_bins_completed(self, times):

        return torch.maximum(
            torch.minimum(
                (times[:, None] - self.bin_starts[None, :]) / self.bin_lengths[None, :],
                torch.tensor(1),
            ),
            torch.tensor(0),
        )

    def forward(self, predictions, event_indicators, event_times):

        event_likelihood = (
            torch.sum(self._discretize_times(event_times) * predictions[:, :-1], -1)
            + self.tolerance
        )

        nonevent_likelihood = (
            1
            - torch.sum(
                self._get_proportion_of_bins_completed(event_times)
                * predictions[:, :-1],
                -1,
            )
            + self.tolerance
        )

        log_likelihood = event_indicators * torch.log(event_likelihood)
        log_likelihood += (1 - event_indicators) * torch.log(nonevent_likelihood)

        return -1.0 * torch.mean(log_likelihood)
