# app/pipeline.py
def process_ticket(row, system):

    issue = row["Issue"]
    subject = row.get("Subject", issue)  # Fallback to Issue if Subject not available
    company = row.get("Company", "unknown")  # Fallback to unknown if Company not available
    
    print(f"Issue: {issue}")
    candidates = system.retriever(issue)
    docs = system.reranker(issue, candidates)
    print(f"Top retrieved doc: {docs[0][0] if docs else 'None'}")

    confidence = system.confidence(docs)
    print(f"Confidence: {confidence:.2f}")
    risk = system.risk(issue)
    print(f"Risk: {risk}")

    decision = system.decide(confidence, risk)
    print(f"Decision: {decision}")

    request_type = system.classify(issue)
    print(f"Request type: {request_type}")
    product_area = system.detect_area(docs)
    print(f"Product area: {product_area}")

    context = "\n".join([d[1] for d in docs])

    if decision == "replied":
        response = system.generate(issue, context)
    else:
        response = "This issue requires human support."

    justification = f"risk={risk}, confidence={confidence:.2f}"

    return {
        "Issue": issue,
        "Subject": subject,
        "Company": company,
        "Response": response,
        "Product Area": product_area,
        "Status": decision,
        "Request Type": request_type
    }