from pydantic import BaseModel, Field
from textwrap import dedent

class AtomicClaims(BaseModel):
    atomic_claims: list[str] = Field(description="A list of atomic claims extracted from the text.")

class RedundantClaimIndices(BaseModel):
    redundant_claim_indices: list[int] = Field(description="indices of the redundant claims.")

interrogator_prompts = {
    "extract_ac_system_prompt": """You are a helpful assistant.""",
    "extract_ac_user_prompt": """Given context and a paragraph of text, deconstruct the text into the smallest possible standalone and self-contained facts without semantic repetition. Each fact should come from the text and must be related to the context.

<Context>{context}</Context>
<Text>{text}</Text>""",

    "extract_ac_user_prompt_strict": """Given context and a paragraph of text, deconstruct the text into the smallest possible standalone and self-contained facts without semantic repetition. Each fact should come from the text and must be related to the context.

<Context>{context}</Context>
<Text>{text}</Text>

Return ONLY a list of facts, with no additional text.
""",

    "q_from_claims_system_prompt": """You are a helpful assistant.""",
    "q_from_claims_user_prompt": """Given context and a list of claims, generate specific, clear questions that have their answers contained in the corresponding claims.
For each claim, generate 1-3 questions that ask for factual information. The generated questions should start from asking the most general information and must be related to the context.

<Context>{context}</Context>
<Claims>
{claims}
</Claims>""",

    "q_from_single_claim_system_prompt": """You are a helpful assistant.""",
    "q_from_single_claim_user_prompt": """Given context and a claim, generate one specific, clear question that has its answer contained in the claim. The generated question must be self-contained and related to the context.
Return only the question, with no additional text.

Context: {context}

Claim: {claim}""",

    "rank_q_system_prompt": """You are a helpful assistant.""",
    "rank_q_user_prompt": """You are given a list of numbered questions. Carefully follow the instructions below to compile the questions into ranked question sets:
1. Cluster similar questions into sets, where each pair of questions in a set are similar to each other. Two questions are similar if they are phrased similarly, or potentially have the same answer.
2. Rerank the question sets by generality, that is, the most general questions should be ranked first, and the most specific, focused questions should be ranked last.
3. Return a list of ranked question sets, where each set is a list of question numbers given in the original question list.

<Question List>
{question_list}
</Question List>""",

    "rm_redundant_ac_system_prompt": """You are a helpful assistant who carefully examines difference between statements.""",
    "rm_redundant_ac_user_prompt_2": """Given two lists of claims, check the redundancy for claims in list B with respect to claims in list A.
For each claim in list B, if it contains information that is already present in any claim in list A, label it as a redundant claim.

<Claim List A>
{claim_list_A}
</Claim List A>

<Claim List B>
{claim_list_B}
</Claim List B>""",

    "rm_redundant_ac_user_prompt": """Given two lists of claims, check the redundancy for claims in list B with respect to claims in list A.
Claim in list B is deemed redundant if it contains information that is already present in any claim in list A. Return a list of indices of the redundant claims in list B.

<Claim List A>
{claim_list_A}
</Claim List A>

<Claim List B>
{claim_list_B}
</Claim List B>""",

    "cluster_similar_ac_system_prompt": """You are a helpful assistant.""",
    "cluster_similar_ac_user_prompt": """You are given a list of statements, where statements are labeled in numerical order. Return sets of labels where each set contains similar statements.

<Statement List>
{statement_list}
</Statement List>""",

    "combine_ac_system_prompt": """You are a helpful assistant.""",
    "combine_ac_user_prompt": """You are given a list of statements, where each statement is similar to each other but not necessarily identical. Remove redundant information and return 1-3 concise and clear statements.

<Statement List>
{statement_list}
</Statement List>""",
}

responder_prompts = {
    "respond": """Answer the following question based on the given context. Format your answer in one sentence:

Context: {context}

Question: {question}

Answer: """,

    "contradiction": """You will be given a statement and a context. Suppose the statement is TRUE, how much of the context will you change to keep it consistent with the statement?
Your final answer should be a percentage number between 0 and 100, representing the percentage of the context you will change.

<Statement>
{statement}
</Statement>

<Context>
{context}
</Context>

Return your answer as a percentage number ONLY, with no additional text.""",

    "contradiction2": """You will be given a statement and a context. Please estimate how much of the context contradicts the statement?
Your final answer should be a percentage number between 0 and 100, representing the percentage of the context that contradicts the statement.

<Statement>
{statement}
</Statement>

<Context>
{context}
</Context>

Return your answer as a percentage number ONLY, with no additional text."""

}

uncertainty_metrics_prompts = {
    "self_consistency_system_prompt": """You are a helpful assistant who carefully examines difference between statements.""",
    "self_consistency_user_prompt": """Given a test claim and a list of candidate claims, rate the level of consistency between the test claim and candidate claims. Pay special attention to factual consistency. Return a score between 0 and 10, representing the level of consistency. Return ONLY the score, nothing else.

<Test Claim>
{test_claim}
</Test Claim>

<Candidate Claims>
{candidate_claims}
</Candidate Claims>""",
}

evaluator_prompts = {
    "eval_claims_from_reference_system_prompt": """You are a meticulous fact-checker who checks the correctness of claims based on reference documents.""",
    "eval_claims_from_reference_user_prompt_single": """Is the following claim correct according to the reference passage? Choose your answer from <correct/incorrect/not_enough_information>.

<Claim>{claim}</Claim>

<Reference>{reference}</Reference>""",

    "from_generations_system_prompt": """You are a meticulous fact-checker who checks the correctness of claims based on a given passage.""",
    "from_generations_user_prompt": """Is the following claim supported by the given passage?

<Claim>
{claim}
</Claim>

<Passage>
{passage}
</Passage>""",

    "from_generations_user_prompt_strict": """Is the following claim supported by the given passage?

<Claim>
{claim}
</Claim>

<Passage>
{passage}
</Passage>

Return "True" if the claim is supported by the passage, return "False" otherwise. Return ONLY the result, nothing else.""",

}