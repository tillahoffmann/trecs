.PHONY : tests

tests :
	pytest tests --cov=spotify_recommender --cov-report=term-missing -v
