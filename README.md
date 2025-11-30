# 🐸 Goumis - Fine-tuning GPT-2 com Greentexts do 4chan

Este projeto realiza o fine-tuning do modelo GPT-2 utilizando um dataset de greentexts coletados do 4chan. 

## 📖 Sobre o Projeto

O **Goumis** é um experimento de aprendizado de máquina que treina o modelo de linguagem GPT-2 da OpenAI para gerar textos no estilo característico das "greentexts" — histórias curtas e humorísticas originadas nos fóruns do 4chan, tipicamente escritas em linhas que começam com `>`.

## 🎯 Objetivo

O objetivo principal é fazer com que o modelo aprenda o estilo único de escrita das greentexts, incluindo:
- Formato de texto com linhas iniciando com `>`
- Narrativa em primeira pessoa
- Tom humorístico e absurdo
- Estrutura típica de "história de anônimo"

## 📦 Dependências

O projeto utiliza as seguintes bibliotecas principais:

- **transformers** (>=4.57.3) - Para carregar e treinar o modelo GPT-2
- **torch** (>=2. 9.1) - Framework de deep learning
- **datasets** (>=4. 4.1) - Para manipulação do dataset
- **tiktoken** (>=0.12.0) - Tokenização
- **tqdm** (>=4.67.1) - Barras de progresso

## 🚀 Instalação

```bash
# Clone o repositório
git clone https://github. com/mnsgrosa/goumis.git
cd goumis

# Instale as dependências usando uv
uv sync

# Ou usando pip
pip install -e .
```

## 📂 Estrutura do Projeto

```
goumis/
├── main.py              # Script principal
├── src/                 # Código fonte do projeto
├── greentext_data/      # Dataset de greentexts
├── log/                 # Logs de treinamento
├── pyproject.toml       # Configurações do projeto
└── README.md            # Este arquivo
```

## 🗃️ Dataset

O dataset utilizado consiste em greentexts coletadas do 4chan.  Greentexts são um formato de postagem característico dos imageboards, onde as linhas começam com o símbolo `>` (que aparece em verde no site original, daí o nome). 

### Características do Dataset:
- Formato de texto único e reconhecível
- Histórias curtas e narrativas
- Conteúdo humorístico e satírico
- Linguagem informal da internet

## 🧠 Sobre o GPT-2

O GPT-2 (Generative Pre-trained Transformer 2) é um modelo de linguagem desenvolvido pela OpenAI. Através do processo de fine-tuning, adaptamos o modelo pré-treinado para gerar textos específicos no estilo greentext.

### Processo de Treinamento:
1. Carregamento do modelo GPT-2 pré-treinado
2. Preparação e tokenização do dataset de greentexts
3. Fine-tuning do modelo com os dados específicos
4.  Avaliação e geração de novos textos

## 📝 Uso

```python
from src import main

# Execute o treinamento
python main.py
```

## ⚠️ Aviso

Este projeto é puramente educacional e experimental. O conteúdo gerado pelo modelo pode refletir o estilo e tom do dataset de treinamento.  Use com responsabilidade. 

## 📄 Licença

Este projeto é de código aberto.  Sinta-se livre para usar, modificar e distribuir. 

## 🤝 Contribuições

Contribuições são bem-vindas!  Sinta-se à vontade para abrir issues ou pull requests.

---

*Feito com 🐸 e muito fine-tuning*
