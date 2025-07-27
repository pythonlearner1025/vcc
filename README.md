# Installation

First setup miniconda. Then run: 

```bash
conda install -c bioconda sra-tools entrez-direct pigz parallel kb-python
```

Make ref folder 
```bash
mkdir ref && cd ref
```

Install required transcript/gene annotations (GTF), cDNA, and Intronic sequences

```bash
# Annotation file
wget ftp://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz

# cDNA FASTA (transcripts)
wget ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz

# Introns FASTA (optional)
wget ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz
```

Build reference
```bash
kb ref \
  -i hg38.idx \
  -g t2g.txt \
  -f1 Homo_sapiens.GRCh38.cdna.all.fa.gz \
  --workflow standard \
  Homo_sapiens.GRCh38.cdna.all.fa.gz \
  Homo_sapiens.GRCh38.110.gtf.gz
```

