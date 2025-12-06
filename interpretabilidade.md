# Interpretabilidade do modelo

Utilizando a biblioteca SHAP podemos ver como cada token esta impactando a predição do
do modelo, a cada token o logit altera para a saida do proximo token, o SHAP consegue
nos dizer a cada passo como o token afeta o logit do modelo.

## Como e feito

Foi criado uma classe Explainer em que e criado objeto Explainer da biblioteca SHAP,
e recebe o tokenizer do modelo usado que no nosso caso e o tiktoken e juntamente o modelo,
esssa classe salva o Explainer para poder reutilizar sem carregar a cara input passado

## O que nos e mostrado?

Quando e passado um input ele ira nos retornar no jupyter notebook um grafico de como
cada token foi influenciado pelo input, com isso e possivel passar o mouse e ver o 
impacto de cada token do input nao apenas o input total.

## Exemplo 

Input:

```
Scientists confirmed the worst possible outcome
```

Output:

```
> be Scientists
> confirm that extraterrestrial life exists.
> accidentally publish on wallstreet journal.
> Investors panic and sell spacex stocks
> scientists buy stocks
> profit off nerds
```

analise do primeiro output:

```
base value: -2.32
fin(input) = -0.78 -> indica a primeira saida "be"
fin(input) = -0.97 -> indica a segunda saida "Scientists"
```

o procedimento se repete mantendo como os pesos estao influenciados

## Explicacao

Shap e responsavel por ver a saida dos logits e ver como cada token do input esta
impactando nessa saida, com isso e possivel ver como o modelo esta interpretando o input e como cada token esta influenciando a saida do modelo.
