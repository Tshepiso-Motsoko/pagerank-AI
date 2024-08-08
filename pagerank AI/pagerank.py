import random

def transition_model(corpus, page, damping_factor):
    """
    Given a corpus of web pages, the function calculates the probability distribution
    of visiting each page from the current page using the Random Surfer Model.

    Parameters:
    - corpus: A dictionary where each key is a page and each value is a set of all pages linked by that page.
    - page: The current page (a key from the corpus dictionary).
    - damping_factor: The damping factor (probability) of choosing one of the links from the current page.

    Returns:
    - A dictionary representing the probability distribution of visiting each page next.
    """
    # Initialize the probability distribution dictionary with equal probability for each page
    prob_dist = {link: (1 - damping_factor) / len(corpus) for link in corpus}

    # Get the pages linked from the current page
    linked_pages = corpus[page]

    # If the current page has no outgoing links, treat it as if it has links to all pages
    if not linked_pages:
        linked_pages = set(corpus.keys())

    # Distribute the damping_factor probability across the linked pages
    for linked_page in linked_pages:
        prob_dist[linked_page] += damping_factor / len(linked_pages)

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Estimates PageRank for each page by generating samples from a Markov Chain.

    Parameters:
    - corpus: A dictionary mapping each page name to a set of all pages linked to by that page.
    - damping_factor: The damping factor used by the transition model.
    - n: The number of samples to generate.

    Returns:
    - A dictionary with each page's estimated PageRank.
    """
    # Initialize a dictionary to keep track of visit counts to each page
    page_visits = {page: 0 for page in corpus}

    # Start by choosing a random page to simulate the random surfer's starting page
    sample = random.choice(list(corpus.keys()))
    page_visits[sample] += 1

    # Sample n times
    for i in range(1, n):
        # Get the transition model for the current page
        current_prob_dist = transition_model(corpus, sample, damping_factor)
        # Choose the next page based on the probability distribution
        sample = random.choices(list(current_prob_dist.keys()), weights=current_prob_dist.values(), k=1)[0]
        # Increment the count for the visited page
        page_visits[sample] += 1

    # Convert visit counts to probabilities (PageRank)
    page_ranks = {page: visit / n for page, visit in page_visits.items()}

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Calculates PageRank for each page using the iterative algorithm until convergence.

    Parameters:
    - corpus: A dictionary mapping each page name to a set of all pages linked to by that page.
    - damping_factor: The damping factor used in the PageRank formula.

    Returns:
    - A dictionary with each page's PageRank.
    """
    # Initialize PageRank for each page: 1 / N where N is the total number of pages
    page_ranks = {page: 1 / len(corpus) for page in corpus}

    # Loop until convergence (change in rank < 0.001)
    convergence_threshold = 0.001
    while True:
        # Track the total change in PageRank across all pages for this iteration
        total_change = 0
        # Calculate new PageRank values based on current values
        new_page_ranks = {}
        for page in page_ranks:
            new_rank = (1 - damping_factor) / len(corpus)
            # Aggregate ranks from pages that link to the current page
            for possible_linker in corpus:
                if page in corpus[possible_linker]:
                    new_rank += damping_factor * (page_ranks[possible_linker] / len(corpus[possible_linker]))
            new_page_ranks[page] = new_rank
            # Update the total change in PageRank
            total_change += abs(new_page_ranks[page] - page_ranks[page])

        # Update page ranks
        page_ranks = new_page_ranks

        # Check if the PageRank values have converged
        if total_change < convergence_threshold * len(corpus):
            break

    return page_ranks

# Example corpus for testing
corpus_example = {
    "1.html": {"2.html", "3.html"},
    "2.html": {"3.html"},
    "3.html": {"2.html"}
}

# Example use of the functions
damping_factor_example = 0.85
samples_example = 10000

# Generate PageRank estimates using sampling
sample_rank = sample_pagerank(corpus_example, damping_factor_example, samples_example)

# Calculate PageRank using the iterative algorithm
iterative_rank = iterate_pagerank(corpus_example, damping_factor_example)

print("Sample PageRank:", sample_rank)
print("Iterative PageRank:", iterative_rank)
