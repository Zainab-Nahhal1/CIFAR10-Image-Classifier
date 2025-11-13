
.PHONY: train test clean

train:
	python -m source.main --epochs 10 --batch-size 64

test:
	pytest -q

clean:
	rm -rf sample/plots cifar10_best.h5 cifar10_trained_model
