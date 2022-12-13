.PHONY: all clean oriented_isotropic nonoriented_isotropic nonoriented_anisotropic

all: oriented_isotropic nonoriented_isotropic nonoriented_anisotropic

oriented_isotropic: logs/ModelDR/oriented_lp2_k17_zeros_s0

nonoriented_isotropic: logs/ResidualParallel/nonoriented_lp2_k17_zeros_s1

nonoriented_anisotropic: logs/ResidualParallel/nonoriented_lp4_k17_zeros_s1

logs/ModelDR/oriented_lp2_k17_zeros_s0: oriented_isotropic.yaml
	nnpf train --config oriented_isotropic.yaml $(options)

logs/ResidualParallel/nonoriented_lp2_k17_zeros_s1: nonoriented_isotropic.yaml
	nnpf train --config nonoriented_isotropic.yaml $(options)

logs/ResidualParallel/nonoriented_lp4_k17_zeros_s1: nonoriented_anisotropic.yaml
	nnpf train --config nonoriented_anisotropic.yaml $(options)

clean:
	rm -rf logs

