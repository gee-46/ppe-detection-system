def evaluate_worker(person_detected,
                    helmet_detected,
                    vest_detected):

    if not person_detected:
        return "No Worker"

    if helmet_detected and vest_detected:
        return "SAFE"

    return "PPE VIOLATION"


# Example Cases

print(
    evaluate_worker(
        True,
        True,
        True
    )
)

print(
    evaluate_worker(
        True,
        False,
        True
    )
)

print(
    evaluate_worker(
        True,
        False,
        False
    )
)
