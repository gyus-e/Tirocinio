from cag import generate, clean_up

def test(model, tokenizer, device, my_cache, origin_len):
    questions = [
        "Who was Napoleon Bonaparte?",
        "Who was Albert Einstein?",
        "What happened after the breakdown of the Treaty of Amiens?",
    ]
    for question in questions:
        input_ids_q = tokenizer(question + "\n", return_tensors="pt").input_ids.to(device)
        gen_ids_q = generate(model, input_ids_q, my_cache)
        answer = tokenizer.decode(gen_ids_q[0], skip_special_tokens=True)
        print("Q:", question)
        print("A:", answer)
        clean_up(my_cache, origin_len)

