## Method – Architecture

### Model B: CLIP Vision Encoder w. LN Fine-tuning + Lightweight Linear Classifier

- Use a **high-capacity pretrained encoder** (e.g., CLIP) as a strong visual representation source  
- **Only adjust LayerNorm parameters** to efficiently shift the feature distribution without full fine-tuning  
- Add a **small linear classifier** to refine the decision boundary  

> Intuition: keep the powerful CLIP backbone almost frozen, slightly adapt it via LayerNorm,  
> then learn a lightweight head for the target forgery-detection task.

#### Losses

- **CE** → trains **class labels** (real vs. fake)  
- **Align** → pulls **same-class features** closer together  
- **Uniform** → keeps the **overall feature distribution** evenly spread over the feature space  

Overall objective:

\[
\mathcal{L} = \mathcal{L}_{\text{Cross-Entropy}} + \alpha \mathcal{L}_{\text{Align}} + \beta \mathcal{L}_{\text{Uniform}}
\]

#### Architecture Sketch

You can optionally add the figure from the slide, for example:

```markdown
![Model B: CLIP Vision Encoder with LN fine-tuning and linear head](./assets/model_b_clip_ln.png)
