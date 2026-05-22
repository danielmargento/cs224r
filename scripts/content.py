"""
scripts/content.py

Generates curriculum content for the finance tutoring RL project:
  - Knowledge graph (16 concepts + prerequisite edges)
  - Item bank: 240 questions partitioned 4 practice + 1 quiz per
    (concept, difficulty) cell -> 192 practice + 48 quiz items
  - Scenario bank: 40 multi-concept word problems for scenario-based reward

The practice/quiz split is deterministic: position 4 (0-indexed) in each cell
is the quiz item; positions 0-3 are practice. This guarantees that:
  - every (concept, difficulty) is represented in both sets
  - the tutor's available action space (practice pool) is decoupled from the
    score-reward evaluation set (quiz pool), so the score reward measures
    held-out generalization rather than memorized items

Outputs written to content/:
  content/knowledge_graph.json
  content/item_bank.json       (includes practice_items + quiz_items lists)
  content/scenarios.json
  content/constants.json

Usage:
  python scripts/content.py
  python scripts/content.py --output_dir content/
"""

import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE GRAPH
# ---------------------------------------------------------------------------

KNOWLEDGE_GRAPH = {
    "budgeting": {
        "prerequisites": [],
        "description": "Creating and managing a personal budget",
        "domain": "foundational",
    },
    "simple_interest": {
        "prerequisites": [],
        "description": "Calculating interest on principal only",
        "domain": "foundational",
    },
    "income_and_taxes": {
        "prerequisites": [],
        "description": "Understanding income types, tax brackets, and withholding",
        "domain": "tax",
    },
    "emergency_fund": {
        "prerequisites": ["budgeting"],
        "description": "Building a 3-6 month expense reserve",
        "domain": "foundational",
    },
    "compound_interest": {
        "prerequisites": ["simple_interest"],
        "description": "Calculating interest on principal plus accumulated interest",
        "domain": "foundational",
    },
    "credit_score": {
        "prerequisites": ["budgeting"],
        "description": "Understanding credit scores and their determinants",
        "domain": "credit",
    },
    "net_worth": {
        "prerequisites": ["budgeting", "emergency_fund"],
        "description": "Calculating and growing net worth over time",
        "domain": "foundational",
    },
    "apr": {
        "prerequisites": ["compound_interest"],
        "description": "Annual Percentage Rate on loans and credit cards",
        "domain": "credit",
    },
    "tax_advantaged_accounts": {
        "prerequisites": ["income_and_taxes"],
        "description": "401(k), IRA, HSA contribution rules and tax benefits",
        "domain": "tax",
    },
    "insurance": {
        "prerequisites": ["budgeting", "net_worth"],
        "description": "Types of insurance, premiums, deductibles, coverage",
        "domain": "tax",
    },
    "amortization": {
        "prerequisites": ["apr"],
        "description": "How loan payments are split between principal and interest",
        "domain": "credit",
    },
    "debt_payoff": {
        "prerequisites": ["apr", "credit_score"],
        "description": "Avalanche and snowball strategies for paying off debt",
        "domain": "credit",
    },
    "investment_basics": {
        "prerequisites": ["compound_interest", "net_worth", "emergency_fund"],
        "description": "Stocks, bonds, diversification, and risk-return tradeoff",
        "domain": "investing",
    },
    "tax_filing": {
        "prerequisites": ["income_and_taxes", "tax_advantaged_accounts"],
        "description": "Filing taxes, deductions, brackets, refunds",
        "domain": "tax",
    },
    "index_funds": {
        "prerequisites": ["investment_basics"],
        "description": "Passive index fund investing and expense ratios",
        "domain": "investing",
    },
    "retirement_planning": {
        "prerequisites": ["investment_basics", "tax_advantaged_accounts", "tax_filing"],
        "description": "Long-term retirement savings strategy and withdrawal planning",
        "domain": "investing",
    },
}

CONCEPTS = list(KNOWLEDGE_GRAPH.keys())
DIFFICULTIES = [1, 2, 3]
ACTIONS = [(c, d) for c in CONCEPTS for d in DIFFICULTIES]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# ---------------------------------------------------------------------------
# 2. ITEM BANK - 16 concepts x 3 difficulties x 5 questions = 240 total
# ---------------------------------------------------------------------------

ITEM_BANK = {

    "budgeting": {
        1: [
            {"id": "budgeting_1_1", "question": "If your monthly income is $3,000 and you spend $2,400, how much can you save?", "answer": "$600", "distractors": ["$400", "$500", "$700"], "p_correct_if_know": 0.95},
            {"id": "budgeting_1_2", "question": "Which of the following is a fixed expense?", "answer": "Monthly rent", "distractors": ["Groceries", "Entertainment", "Clothing"], "p_correct_if_know": 0.93},
            {"id": "budgeting_1_3", "question": "The 50/30/20 rule allocates 20% of income to what?", "answer": "Savings and debt repayment", "distractors": ["Wants", "Needs", "Entertainment"], "p_correct_if_know": 0.92},
            {"id": "budgeting_1_4", "question": "Which of the following is a variable expense?", "answer": "Groceries", "distractors": ["Car loan payment", "Rent", "Internet bill"], "p_correct_if_know": 0.92},
            {"id": "budgeting_1_5", "question": "A budget helps you primarily by:", "answer": "Tracking where your money goes so you can make intentional decisions", "distractors": ["Guaranteeing you will never go into debt", "Automatically investing your savings", "Eliminating all discretionary spending"], "p_correct_if_know": 0.90},
        ],
        2: [
            {"id": "budgeting_2_1", "question": "You earn $4,500/month. Using 50/30/20, how much goes to needs?", "answer": "$2,250", "distractors": ["$1,350", "$900", "$2,000"], "p_correct_if_know": 0.88},
            {"id": "budgeting_2_2", "question": "Which budgeting method assigns every dollar a specific job?", "answer": "Zero-based budgeting", "distractors": ["Envelope method", "Pay-yourself-first", "50/30/20 rule"], "p_correct_if_know": 0.85},
            {"id": "budgeting_2_3", "question": "A budget surplus occurs when income exceeds expenses. What is the best use of a surplus?", "answer": "Build emergency fund or pay off debt", "distractors": ["Increase discretionary spending", "Ignore it", "Buy luxury items"], "p_correct_if_know": 0.87},
            {"id": "budgeting_2_4", "question": "The pay-yourself-first strategy means:", "answer": "Automatically saving a portion of income before spending anything", "distractors": ["Paying your bills before enjoying any leisure", "Earning income from a side job first", "Paying off debt before saving"], "p_correct_if_know": 0.84},
            {"id": "budgeting_2_5", "question": "You spend $200/month on dining out. Under 50/30/20, this falls under which category?", "answer": "Wants (30%)", "distractors": ["Needs (50%)", "Savings (20%)", "Fixed expenses"], "p_correct_if_know": 0.83},
        ],
        3: [
            {"id": "budgeting_3_1", "question": "You have irregular freelance income averaging $5,000/month with $800 variance. How should you budget?", "answer": "Budget based on minimum expected income (~$4,200)", "distractors": ["Budget based on average ($5,000)", "Budget based on maximum income", "Do not budget with irregular income"], "p_correct_if_know": 0.80},
            {"id": "budgeting_3_2", "question": "Which is the correct order of the financial priority stack?", "answer": "Emergency fund - high-interest debt - retirement - other goals", "distractors": ["Retirement - emergency fund - debt - goals", "Debt - goals - retirement - emergency fund", "Goals - debt - emergency fund - retirement"], "p_correct_if_know": 0.78},
            {"id": "budgeting_3_3", "question": "Your take-home is $6,000. Fixed expenses $1,800, debt $900, variable needs $600. How much is discretionary?", "answer": "$2,700", "distractors": ["$3,300", "$2,100", "$1,800"], "p_correct_if_know": 0.75},
            {"id": "budgeting_3_4", "question": "Lifestyle inflation refers to:", "answer": "Increasing spending as income rises, preventing wealth accumulation", "distractors": ["Price increases due to general inflation", "Spending more when prices drop", "Increasing income to match a higher cost of living"], "p_correct_if_know": 0.74},
            {"id": "budgeting_3_5", "question": "You receive a $5,000 bonus. According to sound financial planning, the best allocation is:", "answer": "Top off emergency fund, then split between debt payoff and retirement contributions", "distractors": ["Spend it all - you earned it", "Invest it entirely in crypto", "Put it all in a checking account"], "p_correct_if_know": 0.72},
        ],
    },

    "simple_interest": {
        1: [
            {"id": "simple_interest_1_1", "question": "What is simple interest on $1,000 at 5% for 1 year?", "answer": "$50", "distractors": ["$500", "$5", "$150"], "p_correct_if_know": 0.95},
            {"id": "simple_interest_1_2", "question": "In I = P x r x t, what does P represent?", "answer": "Principal (initial amount)", "distractors": ["Profit", "Payment", "Percentage"], "p_correct_if_know": 0.94},
            {"id": "simple_interest_1_3", "question": "You borrow $500 at 4% simple interest for 2 years. How much interest do you owe?", "answer": "$40", "distractors": ["$20", "$80", "$100"], "p_correct_if_know": 0.92},
            {"id": "simple_interest_1_4", "question": "Simple interest is calculated on:", "answer": "The original principal only", "distractors": ["Principal plus accumulated interest", "The final balance only", "Monthly payments"], "p_correct_if_know": 0.93},
            {"id": "simple_interest_1_5", "question": "A $200 loan at 10% simple interest for 1 year. Total amount owed at end?", "answer": "$220", "distractors": ["$200", "$240", "$210"], "p_correct_if_know": 0.91},
        ],
        2: [
            {"id": "simple_interest_2_1", "question": "A loan of $2,000 at 6% simple interest paid back in 18 months. Total interest?", "answer": "$180", "distractors": ["$240", "$120", "$360"], "p_correct_if_know": 0.85},
            {"id": "simple_interest_2_2", "question": "You invest $800 at simple interest and earn $96 over 2 years. What is the rate?", "answer": "6%", "distractors": ["4%", "8%", "12%"], "p_correct_if_know": 0.83},
            {"id": "simple_interest_2_3", "question": "At what simple interest rate does money double in 10 years?", "answer": "10%", "distractors": ["5%", "20%", "7%"], "p_correct_if_know": 0.80},
            {"id": "simple_interest_2_4", "question": "You need $1,500 in 3 years. A simple interest account pays 5%. How much to deposit now?", "answer": "Approximately $1,304", "distractors": ["$1,500", "$1,275", "$1,425"], "p_correct_if_know": 0.78},
            {"id": "simple_interest_2_5", "question": "A $3,000 loan at 8% simple interest for 2 years. Total repayment amount?", "answer": "$3,480", "distractors": ["$3,240", "$3,600", "$3,000"], "p_correct_if_know": 0.82},
        ],
        3: [
            {"id": "simple_interest_3_1", "question": "You need $5,000 in 3 years. A simple interest account pays 4%. How much to deposit today?", "answer": "Approximately $4,464", "distractors": ["$4,000", "$4,600", "$5,000"], "p_correct_if_know": 0.75},
            {"id": "simple_interest_3_2", "question": "Loan A: $1,000 at 8% simple for 3 years. Loan B: $1,200 at 5% simple for 3 years. Which costs more in interest?", "answer": "Loan A ($240 vs $180)", "distractors": ["Loan B ($180 vs $240)", "They cost the same", "Cannot be determined"], "p_correct_if_know": 0.72},
            {"id": "simple_interest_3_3", "question": "A payday loan charges $15 per $100 for 2 weeks. What is the approximate APR?", "answer": "Approximately 390%", "distractors": ["15%", "180%", "78%"], "p_correct_if_know": 0.70},
            {"id": "simple_interest_3_4", "question": "You earned $270 in simple interest over 3 years on a deposit. The rate was 6%. What was the principal?", "answer": "$1,500", "distractors": ["$900", "$2,700", "$1,350"], "p_correct_if_know": 0.68},
            {"id": "simple_interest_3_5", "question": "Two investments: $5,000 at 7% simple for 4 years vs $4,000 at 9% simple for 4 years. Which earns more interest?", "answer": "They earn the same ($1,400 each)", "distractors": ["$5,000 at 7% earns more", "$4,000 at 9% earns more", "Cannot be determined"], "p_correct_if_know": 0.65},
        ],
    },

    "income_and_taxes": {
        1: [
            {"id": "income_taxes_1_1", "question": "Gross income is:", "answer": "Total income before any deductions or taxes", "distractors": ["Income after taxes are removed", "Only income from your primary job", "Income minus living expenses"], "p_correct_if_know": 0.95},
            {"id": "income_taxes_1_2", "question": "The US tax system is progressive. This means:", "answer": "Higher portions of income are taxed at higher rates", "distractors": ["All income is taxed at your highest bracket rate", "Taxes increase with age", "Everyone pays the same flat rate"], "p_correct_if_know": 0.92},
            {"id": "income_taxes_1_3", "question": "A W-2 form shows:", "answer": "Your wages earned and taxes withheld by your employer", "distractors": ["Your self-employment income", "Your investment gains for the year", "A bill from the IRS"], "p_correct_if_know": 0.90},
            {"id": "income_taxes_1_4", "question": "Net income (take-home pay) is:", "answer": "Gross income minus taxes and deductions", "distractors": ["Gross income plus bonuses", "Income before taxes", "Only your salary, excluding tips"], "p_correct_if_know": 0.93},
            {"id": "income_taxes_1_5", "question": "FICA taxes fund:", "answer": "Social Security and Medicare", "distractors": ["Federal income tax", "State roads and infrastructure", "Unemployment insurance"], "p_correct_if_know": 0.88},
        ],
        2: [
            {"id": "income_taxes_2_1", "question": "You earn $50,000. The 22% bracket starts at $44,725. How much is taxed at 22%?", "answer": "$5,275", "distractors": ["$50,000", "$44,725", "$11,000"], "p_correct_if_know": 0.82},
            {"id": "income_taxes_2_2", "question": "A 1099 form is typically received by:", "answer": "Freelancers and independent contractors", "distractors": ["Full-time employees only", "Anyone with a bank account", "People who receive a tax refund"], "p_correct_if_know": 0.83},
            {"id": "income_taxes_2_3", "question": "Marginal tax rate refers to:", "answer": "The rate applied to your next dollar of income", "distractors": ["The average rate across all your income", "The rate applied to capital gains", "A penalty tax for high earners"], "p_correct_if_know": 0.80},
            {"id": "income_taxes_2_4", "question": "A tax deduction reduces your:", "answer": "Taxable income", "distractors": ["Tax bill dollar for dollar", "Gross income", "FICA contributions"], "p_correct_if_know": 0.81},
            {"id": "income_taxes_2_5", "question": "A tax credit reduces your:", "answer": "Tax bill dollar for dollar", "distractors": ["Taxable income", "Gross income", "State income tax only"], "p_correct_if_know": 0.79},
        ],
        3: [
            {"id": "income_taxes_3_1", "question": "As a freelancer earning $80,000, approximately how much to set aside for quarterly estimated taxes?", "answer": "25-30% (~$20,000-$24,000) to cover income and self-employment taxes", "distractors": ["10-15%", "Nothing if you file annually", "Exactly the same as a W-2 employee"], "p_correct_if_know": 0.72},
            {"id": "income_taxes_3_2", "question": "The self-employment tax rate is approximately 15.3%. Why?", "answer": "You pay both the employee and employer share of Social Security and Medicare", "distractors": ["Freelancers are taxed at a penalty rate", "It includes a state tax component", "It replaces the need to pay income tax"], "p_correct_if_know": 0.70},
            {"id": "income_taxes_3_3", "question": "Your effective tax rate is 18% and marginal rate is 22%. This means:", "answer": "You pay 18% of total income on average, but your last dollars earned are taxed at 22%", "distractors": ["You pay 22% on all income", "You pay 18% on all income", "Your rates are miscalculated"], "p_correct_if_know": 0.68},
            {"id": "income_taxes_3_4", "question": "Passive income from rental properties is generally taxed as:", "answer": "Ordinary income, with some deductions allowed for depreciation and expenses", "distractors": ["Capital gains at a lower rate", "Tax-free if under $10,000", "Always at a flat 15% rate"], "p_correct_if_know": 0.65},
            {"id": "income_taxes_3_5", "question": "You have $5,000 in capital gains and $3,000 in capital losses. How are you taxed?", "answer": "On $2,000 net gains only; losses offset the gains", "distractors": ["On the full $5,000 gains", "No tax - losses cancel everything", "On $5,000 minus the standard deduction"], "p_correct_if_know": 0.67},
        ],
    },

    "emergency_fund": {
        1: [
            {"id": "emergency_fund_1_1", "question": "Financial advisors typically recommend an emergency fund covering how many months of expenses?", "answer": "3-6 months", "distractors": ["1 month", "12 months", "1 week"], "p_correct_if_know": 0.95},
            {"id": "emergency_fund_1_2", "question": "The best place to keep an emergency fund is:", "answer": "High-yield savings account", "distractors": ["Stock market", "Checking account", "Under your mattress"], "p_correct_if_know": 0.90},
            {"id": "emergency_fund_1_3", "question": "An emergency fund is primarily for:", "answer": "Unexpected essential expenses (job loss, medical, car repair)", "distractors": ["Vacations", "Holiday gifts", "Planned purchases"], "p_correct_if_know": 0.93},
            {"id": "emergency_fund_1_4", "question": "Which of these qualifies as an emergency for your emergency fund?", "answer": "Unexpected car repair to get to work", "distractors": ["A sale on a TV you wanted", "A planned vacation", "A birthday gift for a friend"], "p_correct_if_know": 0.92},
            {"id": "emergency_fund_1_5", "question": "Why should an emergency fund NOT be invested in the stock market?", "answer": "Market volatility could force you to sell at a loss when you need the money", "distractors": ["It earns too much interest", "It is illegal to do so", "Stocks are only for retirement accounts"], "p_correct_if_know": 0.88},
        ],
        2: [
            {"id": "emergency_fund_2_1", "question": "Your monthly expenses are $3,200. What is your minimum recommended emergency fund?", "answer": "$9,600", "distractors": ["$3,200", "$19,200", "$6,400"], "p_correct_if_know": 0.85},
            {"id": "emergency_fund_2_2", "question": "A freelancer vs a salaried employee: who needs a larger emergency fund?", "answer": "Freelancer - irregular income makes a larger buffer necessary", "distractors": ["Salaried - more expenses", "They need the same amount", "Neither needs one if they have credit cards"], "p_correct_if_know": 0.83},
            {"id": "emergency_fund_2_3", "question": "You have $500/month to spare. No emergency fund, $8,000 in 6% credit card debt. Priority?", "answer": "Build 1-month emergency fund first, then aggressively pay debt", "distractors": ["Pay all debt first, then build emergency fund", "Split 50/50 between both", "Invest the $500 for better long-term returns"], "p_correct_if_know": 0.80},
            {"id": "emergency_fund_2_4", "question": "After using $4,000 of your emergency fund, your immediate next financial priority should be:", "answer": "Replenish the emergency fund before returning to other goals", "distractors": ["Continue investing as planned", "Pay off all debt immediately", "Reduce retirement contributions"], "p_correct_if_know": 0.82},
            {"id": "emergency_fund_2_5", "question": "Which account type is BEST for an emergency fund?", "answer": "High-yield savings account with no withdrawal penalties", "distractors": ["6-month CD (certificate of deposit)", "Roth IRA", "Brokerage account invested in bonds"], "p_correct_if_know": 0.79},
        ],
        3: [
            {"id": "emergency_fund_3_1", "question": "Your emergency fund is fully funded. You have no employer 401(k) match and an 8% APR personal loan. What next?", "answer": "Pay off the 8% loan (guaranteed 8% return beats most safe investments)", "distractors": ["Max out Roth IRA first", "Invest in index funds first", "Open a taxable brokerage account"], "p_correct_if_know": 0.72},
            {"id": "emergency_fund_3_2", "question": "You are building an emergency fund but your HYSA pays 4.5% while your credit card charges 20%. What should you do?", "answer": "Pay off the credit card first - the 20% cost far outweighs 4.5% savings interest", "distractors": ["Build the emergency fund first regardless", "Split equally between both", "Only make minimum payments and focus on saving"], "p_correct_if_know": 0.70},
            {"id": "emergency_fund_3_3", "question": "You have a stable government job with good disability insurance. Your emergency fund should be:", "answer": "On the lower end (3 months) - income stability reduces the required buffer", "distractors": ["12 months - always maximize", "6 months minimum regardless of job stability", "No emergency fund needed with insurance"], "p_correct_if_know": 0.68},
            {"id": "emergency_fund_3_4", "question": "Opportunity cost of keeping $15,000 in a 1% savings account vs 4.5% HYSA over 5 years is approximately:", "answer": "About $2,600 in lost interest", "distractors": ["$750", "$150", "$5,000"], "p_correct_if_know": 0.65},
            {"id": "emergency_fund_3_5", "question": "You are laid off with 3 months of expenses saved. Monthly costs $3,500, unemployment pays $1,200/month. How long does your fund effectively last?", "answer": "About 4.6 months ($10,500 / $2,300 monthly gap)", "distractors": ["3 months", "6 months", "2 months"], "p_correct_if_know": 0.63},
        ],
    },

    "compound_interest": {
        1: [
            {"id": "compound_interest_1_1", "question": "How does compound interest differ from simple interest?", "answer": "Compound interest earns interest on previously accumulated interest", "distractors": ["Compound interest only applies to loans", "Compound interest is always lower", "They are the same thing"], "p_correct_if_know": 0.93},
            {"id": "compound_interest_1_2", "question": "$1,000 compounded annually at 10% for 2 years gives you approximately:", "answer": "$1,210", "distractors": ["$1,200", "$1,100", "$1,020"], "p_correct_if_know": 0.90},
            {"id": "compound_interest_1_3", "question": "More frequent compounding (monthly vs annually) results in:", "answer": "Higher effective returns", "distractors": ["Lower effective returns", "No difference", "Lower principal"], "p_correct_if_know": 0.88},
            {"id": "compound_interest_1_4", "question": "The formula A = P(1 + r)^t is used for:", "answer": "Compound interest with annual compounding", "distractors": ["Simple interest", "Monthly payment calculations", "Tax calculations"], "p_correct_if_know": 0.86},
            {"id": "compound_interest_1_5", "question": "Starting to invest at 25 vs 35 (same monthly amount, same rate). Who ends up with more at 65?", "answer": "The person who started at 25 - by a very large margin", "distractors": ["The person who started at 35 - they work harder", "They end up with the same amount", "It depends on the interest rate only"], "p_correct_if_know": 0.89},
        ],
        2: [
            {"id": "compound_interest_2_1", "question": "Using the Rule of 72, how long does it take to double money at 8% compounded annually?", "answer": "9 years", "distractors": ["8 years", "12 years", "6 years"], "p_correct_if_know": 0.85},
            {"id": "compound_interest_2_2", "question": "$500 invested at 6% compounded monthly for 5 years. Approximate final value?", "answer": "Approximately $674", "distractors": ["$650", "$600", "$700"], "p_correct_if_know": 0.80},
            {"id": "compound_interest_2_3", "question": "What is the effective annual rate (EAR) of 12% compounded monthly?", "answer": "Approximately 12.68%", "distractors": ["12%", "12.5%", "13%"], "p_correct_if_know": 0.78},
            {"id": "compound_interest_2_4", "question": "Using Rule of 72, at what rate does money double in 6 years?", "answer": "12%", "distractors": ["6%", "8%", "18%"], "p_correct_if_know": 0.82},
            {"id": "compound_interest_2_5", "question": "$1,000 invested at 7% compounded annually for 10 years. Approximate value?", "answer": "Approximately $1,967", "distractors": ["$1,700", "$2,500", "$1,500"], "p_correct_if_know": 0.79},
        ],
        3: [
            {"id": "compound_interest_3_1", "question": "You invest $2,000/year for 30 years at 7% compounded annually. Approximate future value?", "answer": "Approximately $189,000", "distractors": ["$60,000", "$120,000", "$250,000"], "p_correct_if_know": 0.72},
            {"id": "compound_interest_3_2", "question": "Starting at 25 vs 35, investing $200/month at 7%. Approximate difference at 65?", "answer": "Approximately $300,000 more if starting at 25", "distractors": ["$100,000 more", "$500,000 more", "No significant difference"], "p_correct_if_know": 0.70},
            {"id": "compound_interest_3_3", "question": "What nominal rate compounded quarterly equals a 10% effective annual rate?", "answer": "Approximately 9.65%", "distractors": ["10%", "9%", "10.38%"], "p_correct_if_know": 0.68},
            {"id": "compound_interest_3_4", "question": "An investment triples in 18 years. Using Rule of 114, what is the approximate annual rate?", "answer": "Approximately 6.3%", "distractors": ["18%", "3%", "10%"], "p_correct_if_know": 0.65},
            {"id": "compound_interest_3_5", "question": "$10,000 invested at 8% for 20 years compounded monthly vs annually. Approximate difference?", "answer": "About $1,000 more with monthly compounding", "distractors": ["No difference", "$5,000 more", "$500 less"], "p_correct_if_know": 0.63},
        ],
    },

    "credit_score": {
        1: [
            {"id": "credit_score_1_1", "question": "What is the FICO score range?", "answer": "300-850", "distractors": ["0-100", "500-900", "200-800"], "p_correct_if_know": 0.95},
            {"id": "credit_score_1_2", "question": "Which factor has the largest impact on your FICO credit score?", "answer": "Payment history (35%)", "distractors": ["Credit utilization (30%)", "Length of history (15%)", "New credit (10%)"], "p_correct_if_know": 0.90},
            {"id": "credit_score_1_3", "question": "A score above 740 is generally considered:", "answer": "Very good / excellent", "distractors": ["Poor", "Fair", "Good"], "p_correct_if_know": 0.92},
            {"id": "credit_score_1_4", "question": "Credit utilization is:", "answer": "The percentage of available credit you are currently using", "distractors": ["The number of credit cards you own", "How often you apply for new credit", "The age of your oldest account"], "p_correct_if_know": 0.89},
            {"id": "credit_score_1_5", "question": "To build credit from scratch, a good first step is:", "answer": "Open a secured credit card and pay it off monthly", "distractors": ["Take out a personal loan immediately", "Avoid all credit products", "Apply for multiple credit cards at once"], "p_correct_if_know": 0.87},
        ],
        2: [
            {"id": "credit_score_2_1", "question": "Your credit limit is $5,000 and balance is $2,000. What is your utilization rate?", "answer": "40%", "distractors": ["20%", "60%", "25%"], "p_correct_if_know": 0.85},
            {"id": "credit_score_2_2", "question": "Which action will most quickly improve a low credit score?", "answer": "Making on-time payments consistently", "distractors": ["Opening many new accounts", "Closing old accounts", "Applying for multiple loans"], "p_correct_if_know": 0.83},
            {"id": "credit_score_2_3", "question": "A hard inquiry on your credit report typically reduces your score by approximately:", "answer": "5 points or fewer", "distractors": ["50 points", "25 points", "No effect"], "p_correct_if_know": 0.80},
            {"id": "credit_score_2_4", "question": "Closing an old credit card you do not use will likely:", "answer": "Hurt your score by reducing available credit and shortening credit history", "distractors": ["Improve your score by simplifying your profile", "Have no effect on your score", "Improve your score by reducing open accounts"], "p_correct_if_know": 0.78},
            {"id": "credit_score_2_5", "question": "The recommended maximum credit utilization to maintain a good score is:", "answer": "Below 30%", "distractors": ["Below 50%", "Below 10%", "Below 75%"], "p_correct_if_know": 0.82},
        ],
        3: [
            {"id": "credit_score_3_1", "question": "Three cards with limits $2k, $3k, $5k and balances $500, $1,500, $2,000. Total utilization?", "answer": "40% ($4,000 / $10,000)", "distractors": ["33%", "45%", "25%"], "p_correct_if_know": 0.75},
            {"id": "credit_score_3_2", "question": "Which strategy best improves credit score while minimizing interest paid?", "answer": "Pay full balance monthly and keep utilization under 30%", "distractors": ["Carry a small balance each month", "Only make minimum payments", "Close unused cards"], "p_correct_if_know": 0.72},
            {"id": "credit_score_3_3", "question": "You are denied a loan due to a 580 score. Two concrete steps to reach 680 in 12 months?", "answer": "Pay all bills on time + reduce utilization below 30%", "distractors": ["Open new cards + dispute all negatives", "Close old accounts + pay minimums", "Avoid credit entirely + save cash"], "p_correct_if_know": 0.70},
            {"id": "credit_score_3_4", "question": "A $25,000 car loan at 9% APR (620 score) vs 5% APR (740 score) over 5 years. Savings from better score?", "answer": "Approximately $2,800 in total interest savings", "distractors": ["$500", "$10,000", "$1,000"], "p_correct_if_know": 0.68},
            {"id": "credit_score_3_5", "question": "A collection account was paid off 2 years ago but still appears on your report. When does it fall off?", "answer": "After 7 years from the original delinquency date", "distractors": ["Immediately after payoff", "After 2 more years", "After 10 years"], "p_correct_if_know": 0.65},
        ],
    },

    "net_worth": {
        1: [
            {"id": "net_worth_1_1", "question": "Net worth is calculated as:", "answer": "Total assets minus total liabilities", "distractors": ["Total income minus total expenses", "Total savings minus total debt", "Annual salary minus taxes"], "p_correct_if_know": 0.95},
            {"id": "net_worth_1_2", "question": "Which of the following is an asset?", "answer": "A savings account balance", "distractors": ["A credit card balance", "A car loan", "A mortgage balance"], "p_correct_if_know": 0.92},
            {"id": "net_worth_1_3", "question": "Which of the following is a liability?", "answer": "A student loan balance", "distractors": ["A retirement account", "A home you own outright", "Cash in a checking account"], "p_correct_if_know": 0.91},
            {"id": "net_worth_1_4", "question": "You have $15,000 in savings and $8,000 in debt. Your net worth is:", "answer": "$7,000", "distractors": ["$23,000", "$15,000", "-$7,000"], "p_correct_if_know": 0.93},
            {"id": "net_worth_1_5", "question": "A negative net worth means:", "answer": "You owe more than you own", "distractors": ["You have no assets", "You have never invested", "Your income is below average"], "p_correct_if_know": 0.90},
        ],
        2: [
            {"id": "net_worth_2_1", "question": "Your home is worth $300,000 and you owe $200,000 on the mortgage. Home equity contribution to net worth?", "answer": "$100,000", "distractors": ["$300,000", "$200,000", "$500,000"], "p_correct_if_know": 0.85},
            {"id": "net_worth_2_2", "question": "Which action INCREASES net worth?", "answer": "Paying down a car loan", "distractors": ["Taking on a new credit card balance", "Buying a depreciating asset with debt", "Spending your savings on a vacation"], "p_correct_if_know": 0.82},
            {"id": "net_worth_2_3", "question": "You buy a $30,000 car with a $25,000 loan. Immediate impact on net worth?", "answer": "Net +$5,000 immediately (asset $30k - liability $25k), then depreciates over time", "distractors": ["Net worth increases by $30,000", "Net worth is unchanged", "Net worth decreases by $30,000"], "p_correct_if_know": 0.78},
            {"id": "net_worth_2_4", "question": "Tracking net worth monthly helps you:", "answer": "See if your overall financial position is improving over time", "distractors": ["Guarantee investment returns", "Avoid paying taxes", "Qualify for better credit cards"], "p_correct_if_know": 0.83},
            {"id": "net_worth_2_5", "question": "A 401(k) account balance is:", "answer": "An asset that increases your net worth", "distractors": ["A liability until you retire", "Not counted in net worth calculations", "A liability because of future tax owed"], "p_correct_if_know": 0.80},
        ],
        3: [
            {"id": "net_worth_3_1", "question": "Assets: home equity $120k, 401k $80k, car $15k, savings $20k. Liabilities: student loans $25k, credit cards $5k. Net worth?", "answer": "$205,000", "distractors": ["$235,000", "$260,000", "$175,000"], "p_correct_if_know": 0.75},
            {"id": "net_worth_3_2", "question": "Depreciation of a car affects net worth by:", "answer": "Reducing the asset value over time, lowering net worth even if no payments are owed", "distractors": ["Increasing net worth as the loan decreases", "Having no effect if the car is paid off", "Only affecting net worth at point of sale"], "p_correct_if_know": 0.72},
            {"id": "net_worth_3_3", "question": "Why might a high earner have a lower net worth than a modest earner of the same age?", "answer": "Lifestyle inflation and high spending can offset high income; wealth is built by saving the gap", "distractors": ["High earners pay more in taxes, leaving less to save", "High earners have higher liabilities by law", "This is impossible - income always drives net worth"], "p_correct_if_know": 0.70},
            {"id": "net_worth_3_4", "question": "The 4% rule relates to net worth by suggesting:", "answer": "You can retire when net worth reaches 25x your annual expenses", "distractors": ["Save 4% of income each year", "Net worth should grow 4% annually", "Spend no more than 4% of gross income on housing"], "p_correct_if_know": 0.65},
            {"id": "net_worth_3_5", "question": "Income $70,000, expenses $55,000, taxes $10,000. Can you grow net worth by $20,000/year?", "answer": "No - surplus is only $5,000; expense reduction needed to reach $20,000 savings", "distractors": ["Yes - straightforward with current numbers", "No - impossible on this income", "Yes - invest the $15k surplus"], "p_correct_if_know": 0.68},
        ],
    },

    "apr": {
        1: [
            {"id": "apr_1_1", "question": "APR stands for:", "answer": "Annual Percentage Rate", "distractors": ["Applied Payment Rate", "Annual Principal Ratio", "Asset Purchase Rate"], "p_correct_if_know": 0.95},
            {"id": "apr_1_2", "question": "A credit card has 24% APR. What is the approximate monthly rate?", "answer": "2%", "distractors": ["24%", "0.2%", "12%"], "p_correct_if_know": 0.90},
            {"id": "apr_1_3", "question": "A lower APR on a loan means:", "answer": "Less interest paid over time", "distractors": ["Higher monthly payments", "Longer loan term", "More fees"], "p_correct_if_know": 0.92},
            {"id": "apr_1_4", "question": "APR includes:", "answer": "Interest rate plus fees, expressed as a yearly rate", "distractors": ["Only the interest rate", "Only the origination fees", "The monthly payment amount"], "p_correct_if_know": 0.88},
            {"id": "apr_1_5", "question": "Which APR is better for a borrower?", "answer": "6% APR", "distractors": ["18% APR", "12% APR", "24% APR"], "p_correct_if_know": 0.95},
        ],
        2: [
            {"id": "apr_2_1", "question": "A $10,000 loan at 8% APR for 3 years. Approximate total interest paid?", "answer": "Approximately $1,300", "distractors": ["$800", "$2,400", "$3,000"], "p_correct_if_know": 0.82},
            {"id": "apr_2_2", "question": "What is the difference between APR and APY?", "answer": "APY includes compounding effects; APR does not", "distractors": ["APR includes fees; APY does not", "They are the same", "APY is always lower than APR"], "p_correct_if_know": 0.80},
            {"id": "apr_2_3", "question": "You carry a $2,000 credit card balance at 18% APR. Monthly interest charge?", "answer": "$30", "distractors": ["$360", "$18", "$60"], "p_correct_if_know": 0.83},
            {"id": "apr_2_4", "question": "A 0% APR promo for 12 months has a 3% balance transfer fee. Worth it for $5,000 at 22% APR?", "answer": "Yes - saves approximately $950 in interest vs $150 fee", "distractors": ["No - fees always outweigh benefits", "Only if you pay off in 6 months", "Cannot be determined without credit score"], "p_correct_if_know": 0.78},
            {"id": "apr_2_5", "question": "Your savings account pays 4.5% APY. A loan costs 6% APR. Should you use savings to pay off the loan?", "answer": "Yes - the 6% cost exceeds the 4.5% gain; paying off is the better return", "distractors": ["No - always keep savings intact", "No - APY and APR are not comparable", "Only if the loan has no prepayment penalty"], "p_correct_if_know": 0.75},
        ],
        3: [
            {"id": "apr_3_1", "question": "Loan A: $15,000 at 6% APR for 5 years. Loan B: $15,000 at 5% APR for 7 years. Which has lower total cost?", "answer": "Loan A - shorter term wins despite higher rate", "distractors": ["Loan B", "They are equal", "Cannot be determined"], "p_correct_if_know": 0.72},
            {"id": "apr_3_2", "question": "What effective annual rate corresponds to 18% APR compounded monthly?", "answer": "Approximately 19.56%", "distractors": ["18%", "19%", "20%"], "p_correct_if_know": 0.68},
            {"id": "apr_3_3", "question": "A store offers 0% APR for 24 months but inflates the item price by 15%. Is this a good deal?", "answer": "Probably not - the 15% price markup likely costs more than 24 months of moderate interest", "distractors": ["Yes - 0% is always the best financing", "Yes - inflation makes paying later better", "Depends only on your credit score"], "p_correct_if_know": 0.65},
            {"id": "apr_3_4", "question": "Card A $3,000 at 22% APR, Card B $1,000 at 15% APR. Avalanche method says:", "answer": "Pay minimums on both, put extra money toward Card A (higher rate)", "distractors": ["Pay off Card B first (smaller balance)", "Pay equal amounts to both", "Only pay Card A"], "p_correct_if_know": 0.70},
            {"id": "apr_3_5", "question": "A mortgage APR of 7.2% vs interest rate of 7%. The difference represents:", "answer": "Fees and closing costs rolled into the APR making the true cost of borrowing higher", "distractors": ["A calculation error", "State taxes on the loan", "The lender profit margin only"], "p_correct_if_know": 0.67},
        ],
    },

    "tax_advantaged_accounts": {
        1: [
            {"id": "tax_adv_1_1", "question": "A traditional 401(k) contribution reduces your:", "answer": "Taxable income now", "distractors": ["Taxable income at withdrawal", "Social Security taxes", "Medicare taxes"], "p_correct_if_know": 0.90},
            {"id": "tax_adv_1_2", "question": "A Roth IRA uses after-tax dollars. Growth and qualified withdrawals are:", "answer": "Tax-free", "distractors": ["Taxed as ordinary income", "Taxed at capital gains rate", "Partially taxed"], "p_correct_if_know": 0.88},
            {"id": "tax_adv_1_3", "question": "An HSA (Health Savings Account) requires enrollment in:", "answer": "A high-deductible health plan (HDHP)", "distractors": ["Any health plan", "Medicare", "Medicaid"], "p_correct_if_know": 0.85},
            {"id": "tax_adv_1_4", "question": "The main advantage of a 401(k) employer match is:", "answer": "It is free money added to your retirement savings", "distractors": ["It reduces your taxes by double", "It guarantees investment returns", "It lowers your health insurance premiums"], "p_correct_if_know": 0.91},
            {"id": "tax_adv_1_5", "question": "What does tax-deferred mean in the context of a traditional IRA?", "answer": "You pay taxes when you withdraw the money in retirement, not when you contribute", "distractors": ["You never pay taxes on the money", "You pay taxes upfront, then withdraw tax-free", "The account is exempt from all taxes"], "p_correct_if_know": 0.87},
        ],
        2: [
            {"id": "tax_adv_2_1", "question": "Employer matches 50% of 401(k) contributions up to 6% of salary. You earn $60,000. Max match per year?", "answer": "$1,800", "distractors": ["$3,600", "$900", "$3,000"], "p_correct_if_know": 0.80},
            {"id": "tax_adv_2_2", "question": "Traditional IRA vs Roth IRA: when is Roth generally preferable?", "answer": "When you expect to be in a higher tax bracket at retirement", "distractors": ["When you expect a lower tax bracket at retirement", "Always - Roth is always better", "When you need a tax deduction now"], "p_correct_if_know": 0.78},
            {"id": "tax_adv_2_3", "question": "The HSA triple tax advantage means:", "answer": "Contributions pre-tax, growth tax-free, qualified withdrawals tax-free", "distractors": ["Three separate accounts with tax benefits", "Contributions, earnings, and employer match all untaxed", "You get three times the standard deduction"], "p_correct_if_know": 0.75},
            {"id": "tax_adv_2_4", "question": "Early withdrawal from a traditional 401(k) before age 59.5 typically incurs:", "answer": "Income taxes plus a 10% penalty", "distractors": ["No penalty if under $10,000", "A flat 20% penalty only", "Capital gains tax only"], "p_correct_if_know": 0.80},
            {"id": "tax_adv_2_5", "question": "Required Minimum Distributions (RMDs) apply to:", "answer": "Traditional 401(k) and IRA accounts starting at age 73", "distractors": ["Roth IRAs at age 65", "All accounts starting at age 59.5", "Only 401(k) accounts, not IRAs"], "p_correct_if_know": 0.72},
        ],
        3: [
            {"id": "tax_adv_3_1", "question": "You are 28, in the 22% bracket, expect to retire in the 24% bracket. Traditional or Roth 401(k)?", "answer": "Roth - pay taxes now at 22% vs later at 24%", "distractors": ["Traditional - always take the deduction now", "Split 50/50 regardless", "Neither - use a taxable brokerage"], "p_correct_if_know": 0.72},
            {"id": "tax_adv_3_2", "question": "What is a backdoor Roth IRA and who uses it?", "answer": "Contributing to a traditional IRA then converting to Roth - used by high earners above Roth income limits", "distractors": ["An illegal tax shelter", "A Roth for people under 18", "A rollover from a 401(k) to a Roth"], "p_correct_if_know": 0.65},
            {"id": "tax_adv_3_3", "question": "Correct order for maximizing tax-advantaged savings (with employer match):", "answer": "401(k) up to match - HSA max - Roth IRA max - 401(k) max", "distractors": ["Roth IRA first - 401(k) - HSA", "401(k) max first - then Roth IRA", "HSA first - Roth IRA - 401(k) match"], "p_correct_if_know": 0.62},
            {"id": "tax_adv_3_4", "question": "You max your 401(k) ($23,000) and Roth IRA ($7,000) at 30. At 7% growth, approximate value at 65?", "answer": "Approximately $4.2M", "distractors": ["$1.05M", "$2.1M", "$8.4M"], "p_correct_if_know": 0.68},
            {"id": "tax_adv_3_5", "question": "A mega backdoor Roth allows after-tax 401(k) contributions up to approximately:", "answer": "$43,500 (total limit $66,000 minus pre-tax contributions)", "distractors": ["$7,000 (same as Roth IRA limit)", "$23,000 (standard 401k limit)", "$100,000 flat"], "p_correct_if_know": 0.58},
        ],
    },

    "insurance": {
        1: [
            {"id": "insurance_1_1", "question": "A deductible in an insurance policy is:", "answer": "The amount you pay out-of-pocket before insurance kicks in", "distractors": ["Your monthly insurance payment", "The maximum the insurer will pay", "A penalty for filing a claim"], "p_correct_if_know": 0.92},
            {"id": "insurance_1_2", "question": "A higher deductible usually means:", "answer": "Lower monthly premiums", "distractors": ["Higher monthly premiums", "Better coverage", "No change in premiums"], "p_correct_if_know": 0.90},
            {"id": "insurance_1_3", "question": "Term life insurance provides coverage:", "answer": "For a specified period (e.g. 20 years)", "distractors": ["For your entire life", "Only for work-related incidents", "Until you stop paying premiums (whole life)"], "p_correct_if_know": 0.88},
            {"id": "insurance_1_4", "question": "An out-of-pocket maximum on a health plan means:", "answer": "After reaching this amount, insurance covers 100% of covered costs", "distractors": ["The most the insurer will spend on your care", "Your total annual premium", "The deductible plus copays only"], "p_correct_if_know": 0.85},
            {"id": "insurance_1_5", "question": "Renters insurance protects:", "answer": "Your personal belongings inside a rented property", "distractors": ["The building structure", "The landlord's property", "Only electronics and jewelry"], "p_correct_if_know": 0.87},
        ],
        2: [
            {"id": "insurance_2_1", "question": "Which type of life insurance builds cash value over time?", "answer": "Whole (permanent) life insurance", "distractors": ["Term life", "Disability insurance", "Liability insurance"], "p_correct_if_know": 0.80},
            {"id": "insurance_2_2", "question": "For most young, healthy adults, financial advisors recommend:", "answer": "Term life + investing the premium difference over whole life", "distractors": ["Whole life for the investment component", "No life insurance until age 40", "Universal life for flexibility"], "p_correct_if_know": 0.78},
            {"id": "insurance_2_3", "question": "An HMO vs PPO health plan: which is better for someone who travels frequently?", "answer": "PPO - allows out-of-network care without referrals", "distractors": ["HMO - lower premiums and no deductible", "Both are equivalent for travelers", "HDHP - best for all situations"], "p_correct_if_know": 0.75},
            {"id": "insurance_2_4", "question": "Disability insurance typically replaces what percentage of income?", "answer": "60-70% of pre-disability income", "distractors": ["100%", "30-40%", "90%"], "p_correct_if_know": 0.72},
            {"id": "insurance_2_5", "question": "Umbrella insurance provides:", "answer": "Liability coverage beyond your home and auto policy limits", "distractors": ["Coverage for all natural disasters", "Life insurance for the whole family", "Business liability protection only"], "p_correct_if_know": 0.70},
        ],
        3: [
            {"id": "insurance_3_1", "question": "You are 28, healthy, no dependents. Which insurance types are most critical?", "answer": "Health + renters/auto + disability (income protection with no safety net)", "distractors": ["Life + long-term care + dental", "Life + health only", "All types equally important"], "p_correct_if_know": 0.72},
            {"id": "insurance_3_2", "question": "You have a $1,000 health deductible and $3,000 out-of-pocket max. A procedure costs $8,000. You pay:", "answer": "$3,000 (you hit the out-of-pocket max)", "distractors": ["$1,000", "$8,000", "$5,000"], "p_correct_if_know": 0.68},
            {"id": "insurance_3_3", "question": "Self-insuring means:", "answer": "Setting aside your own funds to cover potential losses instead of buying insurance", "distractors": ["Buying insurance from yourself", "Waiving all coverage intentionally", "A type of government insurance program"], "p_correct_if_know": 0.65},
            {"id": "insurance_3_4", "question": "A $250 vs $1,000 deductible on auto insurance saves $300/year in premiums. Break-even analysis says:", "answer": "Choose $1,000 deductible if you can cover it - you break even after 2.5 accident-free years", "distractors": ["Always choose the lower deductible", "The $250 deductible always wins", "Choose based on credit score, not savings"], "p_correct_if_know": 0.62},
            {"id": "insurance_3_5", "question": "Life insurance need is primarily determined by:", "answer": "Number of dependents, income replacement needs, and existing assets", "distractors": ["Your age only", "Your net worth only", "A standard multiplier of 10x salary always"], "p_correct_if_know": 0.65},
        ],
    },

    "amortization": {
        1: [
            {"id": "amortization_1_1", "question": "In an amortizing loan, early payments consist mostly of:", "answer": "Interest", "distractors": ["Principal", "Fees", "Equal principal and interest"], "p_correct_if_know": 0.92},
            {"id": "amortization_1_2", "question": "As an amortizing loan matures, the principal portion of each payment:", "answer": "Increases over time", "distractors": ["Decreases", "Stays the same", "Goes to zero"], "p_correct_if_know": 0.90},
            {"id": "amortization_1_3", "question": "An amortization schedule shows:", "answer": "How each payment splits between principal and interest over time", "distractors": ["Only the total interest paid", "The loan origination fees", "Your credit score history"], "p_correct_if_know": 0.91},
            {"id": "amortization_1_4", "question": "A fully amortizing loan means:", "answer": "Equal payments over the loan term result in a zero balance at the end", "distractors": ["You pay interest only", "The balance increases over time", "Payments vary each month"], "p_correct_if_know": 0.88},
            {"id": "amortization_1_5", "question": "Making extra principal payments on a loan will:", "answer": "Reduce the total interest paid and shorten the loan term", "distractors": ["Lower your monthly payment immediately", "Increase your interest rate", "Have no effect on total interest"], "p_correct_if_know": 0.87},
        ],
        2: [
            {"id": "amortization_2_1", "question": "A $200,000 mortgage at 6% for 30 years (~$1,199/month). In month 1, how much goes to interest?", "answer": "Approximately $1,000", "distractors": ["$199", "$600", "$1,199"], "p_correct_if_know": 0.80},
            {"id": "amortization_2_2", "question": "Making one extra principal payment per year on a 30-year mortgage typically:", "answer": "Reduces the loan term by several years", "distractors": ["Has no meaningful impact", "Increases your monthly payment", "Resets the amortization schedule"], "p_correct_if_know": 0.78},
            {"id": "amortization_2_3", "question": "A 15-year vs 30-year mortgage at the same rate: the 15-year has higher monthly payments but:", "answer": "Much lower total interest paid", "distractors": ["Higher total interest paid", "Same total interest", "Lower principal"], "p_correct_if_know": 0.82},
            {"id": "amortization_2_4", "question": "After 5 years on a $300,000/7%/30-year mortgage, what percentage of principal have you paid off?", "answer": "Approximately 7%", "distractors": ["17%", "33%", "50%"], "p_correct_if_know": 0.72},
            {"id": "amortization_2_5", "question": "A balloon mortgage means:", "answer": "Small payments during the term with a large lump sum due at the end", "distractors": ["Payments that increase each year", "A mortgage with no interest", "Payments that decrease over time"], "p_correct_if_know": 0.75},
        ],
        3: [
            {"id": "amortization_3_1", "question": "$300,000 mortgage at 7%, 30 years (~$1,996/month). Total interest over life of loan?", "answer": "Approximately $418,000", "distractors": ["$210,000", "$300,000", "$600,000"], "p_correct_if_know": 0.72},
            {"id": "amortization_3_2", "question": "You refinance a $250,000 mortgage from 7% to 5% with 20 years remaining, paying $3,000 in closing costs. Break-even horizon?", "answer": "Approximately 18 months", "distractors": ["5 years", "6 months", "Never"], "p_correct_if_know": 0.68},
            {"id": "amortization_3_3", "question": "Interest-only loan for 5 years then converts to fully amortizing. The main risk is:", "answer": "Payment shock when principal payments begin; no equity built during interest-only period", "distractors": ["Rate will always increase after conversion", "You lose the house after 5 years", "No risk - payments just decrease"], "p_correct_if_know": 0.65},
            {"id": "amortization_3_4", "question": "A $400,000 mortgage at 6.5% for 30 years. How much interest do you pay in year 1 vs year 29?", "answer": "Year 1: ~$25,800 in interest vs year 29: ~$1,500 - heavily front-loaded", "distractors": ["Equal amounts each year", "More in year 29 as balance grows", "Year 1: $10,000 vs year 29: $20,000"], "p_correct_if_know": 0.62},
            {"id": "amortization_3_5", "question": "Biweekly mortgage payments vs monthly result in:", "answer": "One extra full payment per year, saving interest and cutting years off the loan", "distractors": ["The same total annual payment", "Higher interest due to more transactions", "Lower principal only in the first year"], "p_correct_if_know": 0.65},
        ],
    },

    "debt_payoff": {
        1: [
            {"id": "debt_payoff_1_1", "question": "The debt avalanche method prioritizes:", "answer": "Paying off the highest interest rate debt first", "distractors": ["Paying off the smallest balance first", "Paying equal amounts to all debts", "Paying off the newest debt first"], "p_correct_if_know": 0.92},
            {"id": "debt_payoff_1_2", "question": "The debt snowball method prioritizes:", "answer": "Paying off the smallest balance first regardless of interest rate", "distractors": ["Paying off the highest rate first", "Paying minimums on all debts", "Paying off secured debt first"], "p_correct_if_know": 0.91},
            {"id": "debt_payoff_1_3", "question": "Which debt payoff method saves the most money mathematically?", "answer": "Avalanche (highest rate first)", "distractors": ["Snowball (smallest balance first)", "They save the same amount", "Depends on the number of debts"], "p_correct_if_know": 0.88},
            {"id": "debt_payoff_1_4", "question": "The snowball method is preferred by some because:", "answer": "Quick wins from paying off small debts provide motivation to continue", "distractors": ["It saves the most money", "It has lower interest rates", "Banks prefer it"], "p_correct_if_know": 0.87},
            {"id": "debt_payoff_1_5", "question": "Minimum payments on a credit card primarily pay:", "answer": "Mostly interest, with very little going to principal", "distractors": ["Equal principal and interest", "Mostly principal", "Fees and penalties only"], "p_correct_if_know": 0.89},
        ],
        2: [
            {"id": "debt_payoff_2_1", "question": "Debts: $5,000 at 22%, $2,000 at 15%, $8,000 at 6%. Avalanche order?", "answer": "$5,000 at 22% first, then $2,000 at 15%, then $8,000 at 6%", "distractors": ["$2,000 at 15% first (smallest balance)", "$8,000 at 6% first (largest balance)", "All simultaneously"], "p_correct_if_know": 0.85},
            {"id": "debt_payoff_2_2", "question": "Debt consolidation makes sense when:", "answer": "The new consolidated loan has a lower rate than your existing debts", "distractors": ["You want to extend your repayment period to lower payments", "You have perfect credit", "You have only one type of debt"], "p_correct_if_know": 0.80},
            {"id": "debt_payoff_2_3", "question": "A $3,000 balance at 20% APR with minimum payments of 2%. How long to pay off making only minimums?", "answer": "Over 20 years, paying nearly double the original balance in interest", "distractors": ["About 3 years", "About 7 years", "About 12 years"], "p_correct_if_know": 0.75},
            {"id": "debt_payoff_2_4", "question": "Balance transfer cards offer 0% APR for 12-18 months. The main risk is:", "answer": "If balance is not paid off in time, deferred interest may be applied retroactively", "distractors": ["Your credit score drops permanently", "The original debt increases", "You cannot use the card for purchases"], "p_correct_if_know": 0.72},
            {"id": "debt_payoff_2_5", "question": "A student loan for a high-earning degree is considered:", "answer": "Potentially good debt - investment with positive expected ROI", "distractors": ["Always bad - all debt is harmful", "Good debt only if under $10,000", "Bad debt because it has interest"], "p_correct_if_know": 0.78},
        ],
        3: [
            {"id": "debt_payoff_3_1", "question": "$500/month extra. Avalanche vs snowball on $1,200 at 8% and $4,000 at 19%. Which saves more?", "answer": "Avalanche saves more - targeting the 19% debt first minimizes total interest", "distractors": ["Snowball saves more", "They save the same amount", "Depends on how long you stick with it"], "p_correct_if_know": 0.72},
            {"id": "debt_payoff_3_2", "question": "Debt-to-income (DTI) ratio of 45% means:", "answer": "45% of gross monthly income goes to debt payments - above the typical 36% threshold lenders prefer", "distractors": ["You owe 45% of your annual income", "45% of your net worth is debt", "Your debt grows at 45% per year"], "p_correct_if_know": 0.68},
            {"id": "debt_payoff_3_3", "question": "Should you pay off a 3% mortgage early instead of investing in index funds averaging 8%?", "answer": "Generally no - the expected 8% return exceeds the 3% debt cost; invest instead", "distractors": ["Yes - always pay off debt first", "Yes - guaranteed return is always better", "Only if you have no emergency fund"], "p_correct_if_know": 0.70},
            {"id": "debt_payoff_3_4", "question": "Student loan forgiveness via PSLF requires:", "answer": "10 years of qualifying payments while working full-time for a qualifying public employer", "distractors": ["5 years of payments regardless of employer", "Earning under $50,000 annually", "Graduating from a public university"], "p_correct_if_know": 0.65},
            {"id": "debt_payoff_3_5", "question": "You have $20,000 in savings earning 4.5% and $15,000 in credit card debt at 21%. Best strategy?", "answer": "Pay off the credit card with savings - 21% cost vs 4.5% gain is a 16.5% guaranteed return", "distractors": ["Keep savings intact for emergencies, pay minimums", "Invest the savings in stocks instead", "Use savings to pay off only half"], "p_correct_if_know": 0.68},
        ],
    },

    "investment_basics": {
        1: [
            {"id": "investment_1_1", "question": "Diversification in investing means:", "answer": "Spreading investments across different assets to reduce risk", "distractors": ["Putting all money in the best-performing asset", "Only investing in safe assets", "Investing in multiple accounts at the same bank"], "p_correct_if_know": 0.92},
            {"id": "investment_1_2", "question": "Which asset class is generally considered highest risk / highest potential return?", "answer": "Stocks (equities)", "distractors": ["Government bonds", "CDs", "Money market funds"], "p_correct_if_know": 0.90},
            {"id": "investment_1_3", "question": "A bond is essentially:", "answer": "A loan you make to a company or government in exchange for interest", "distractors": ["Ownership stake in a company", "A savings account with a bank", "A type of insurance policy"], "p_correct_if_know": 0.88},
            {"id": "investment_1_4", "question": "Dollar-cost averaging means:", "answer": "Investing fixed amounts at regular intervals regardless of price", "distractors": ["Buying only when prices are low", "Averaging your cost basis across brokers", "Investing a lump sum once per year"], "p_correct_if_know": 0.85},
            {"id": "investment_1_5", "question": "The risk-return tradeoff means:", "answer": "Higher potential returns require accepting higher risk", "distractors": ["Higher returns are always achievable with patience", "Risk and return are unrelated", "Lower risk means lower returns only in the short term"], "p_correct_if_know": 0.87},
        ],
        2: [
            {"id": "investment_2_1", "question": "Inflation risk in investing refers to:", "answer": "Returns failing to keep pace with inflation, reducing purchasing power", "distractors": ["The government taxing investment gains", "Your broker charging inflation-linked fees", "Interest rates rising"], "p_correct_if_know": 0.82},
            {"id": "investment_2_2", "question": "At age 30 with a 35-year horizon, a reasonable stock/bond allocation is:", "answer": "80-90% stocks, 10-20% bonds", "distractors": ["50% stocks, 50% bonds", "100% bonds", "30% stocks, 70% bonds"], "p_correct_if_know": 0.80},
            {"id": "investment_2_3", "question": "A mutual fund differs from an ETF primarily in that:", "answer": "Mutual funds trade once per day at NAV; ETFs trade throughout the day like stocks", "distractors": ["Mutual funds are always actively managed", "ETFs have higher minimum investments", "Mutual funds never charge fees"], "p_correct_if_know": 0.75},
            {"id": "investment_2_4", "question": "Rebalancing a portfolio means:", "answer": "Adjusting holdings back to your target allocation after market movements", "distractors": ["Selling all investments annually", "Moving to safer assets as you age", "Increasing investments when markets rise"], "p_correct_if_know": 0.78},
            {"id": "investment_2_5", "question": "Which best describes sequence-of-returns risk?", "answer": "Poor returns early in retirement deplete the portfolio before recovery is possible", "distractors": ["The risk that inflation outpaces returns", "The risk of investing in sequence rather than lump sum", "The risk of rebalancing too frequently"], "p_correct_if_know": 0.72},
        ],
        3: [
            {"id": "investment_3_1", "question": "The 4% withdrawal rule assumes:", "answer": "A 30-year retirement with historical market returns sustains the portfolio", "distractors": ["Withdraw 4% monthly for safety", "Invest 4% of income annually", "Keep 4% in cash at all times"], "p_correct_if_know": 0.68},
            {"id": "investment_3_2", "question": "Tax-loss harvesting involves:", "answer": "Selling losing investments to realize a tax loss, then buying a similar (not identical) investment", "distractors": ["Selling winners to pay taxes, then reinvesting", "Avoiding all taxable events permanently", "Holding all investments in tax-advantaged accounts only"], "p_correct_if_know": 0.65},
            {"id": "investment_3_3", "question": "A 100% stock portfolio vs 80/20 stock/bond over 30 years. The 80/20 portfolio likely:", "answer": "Has lower volatility and slightly lower returns - better for risk-averse investors", "distractors": ["Always underperforms 100% stocks", "Always outperforms 100% stocks", "Has identical returns with same risk"], "p_correct_if_know": 0.65},
            {"id": "investment_3_4", "question": "During a market crash, dollar-cost averaging into index funds is advantageous because:", "answer": "You buy more shares at lower prices, lowering your average cost basis", "distractors": ["It protects you from any further losses", "Your existing shares gain value", "It has no advantage - stop investing in crashes"], "p_correct_if_know": 0.68},
            {"id": "investment_3_5", "question": "Factor investing (value, momentum, quality) attempts to:", "answer": "Systematically capture return premiums associated with specific stock characteristics", "distractors": ["Time the market using economic indicators", "Invest only in the top-performing sectors", "Replicate total market returns at lower cost"], "p_correct_if_know": 0.58},
        ],
    },

    "tax_filing": {
        1: [
            {"id": "tax_filing_1_1", "question": "The standard deduction for a single filer in 2024 is approximately:", "answer": "$14,600", "distractors": ["$7,300", "$20,000", "$12,000"], "p_correct_if_know": 0.88},
            {"id": "tax_filing_1_2", "question": "A tax refund means:", "answer": "You overpaid taxes during the year via withholding", "distractors": ["The government owes you extra money", "You get free money back", "You underpaid taxes"], "p_correct_if_know": 0.85},
            {"id": "tax_filing_1_3", "question": "Itemizing deductions is beneficial when:", "answer": "Your itemized deductions exceed the standard deduction", "distractors": ["Always - itemizing always saves more", "Your income exceeds $100,000", "You have a mortgage"], "p_correct_if_know": 0.83},
            {"id": "tax_filing_1_4", "question": "The tax filing deadline in the US is typically:", "answer": "April 15", "distractors": ["March 15", "June 15", "January 31"], "p_correct_if_know": 0.92},
            {"id": "tax_filing_1_5", "question": "Filing an extension gives you more time to:", "answer": "Submit your return, not to pay taxes owed", "distractors": ["Pay any taxes you owe", "Both file and pay", "Avoid any penalties permanently"], "p_correct_if_know": 0.80},
        ],
        2: [
            {"id": "tax_filing_2_1", "question": "You earn $50,000. The 22% bracket starts at $44,725. How much is taxed at 22%?", "answer": "$5,275", "distractors": ["$50,000", "$44,725", "$11,000"], "p_correct_if_know": 0.78},
            {"id": "tax_filing_2_2", "question": "Capital gains tax on assets held more than one year is generally:", "answer": "Lower than ordinary income tax rates (0%, 15%, or 20%)", "distractors": ["The same as ordinary income tax", "Always 28%", "Exempt from federal tax"], "p_correct_if_know": 0.78},
            {"id": "tax_filing_2_3", "question": "The Earned Income Tax Credit (EITC) is designed for:", "answer": "Low to moderate income workers, providing a refundable credit", "distractors": ["High income investors", "People with capital gains", "Retirees only"], "p_correct_if_know": 0.75},
            {"id": "tax_filing_2_4", "question": "Qualified dividends are taxed at:", "answer": "The lower long-term capital gains rate (0%, 15%, or 20%)", "distractors": ["Ordinary income tax rates", "A flat 10%", "They are always tax-free"], "p_correct_if_know": 0.72},
            {"id": "tax_filing_2_5", "question": "The child tax credit in 2024 provides up to how much per qualifying child?", "answer": "Up to $2,000 per child", "distractors": ["$500", "$5,000", "$1,000"], "p_correct_if_know": 0.73},
        ],
        3: [
            {"id": "tax_filing_3_1", "question": "You have $5,000 in capital gains and $3,000 in capital losses. How are you taxed?", "answer": "On $2,000 net gains only; $3,000 loss offsets the gains", "distractors": ["On the full $5,000 gains", "No tax - losses cancel everything", "On $5,000 minus the standard deduction"], "p_correct_if_know": 0.72},
            {"id": "tax_filing_3_2", "question": "The wash-sale rule prevents:", "answer": "Claiming a tax loss if you buy the same or substantially identical security within 30 days", "distractors": ["Selling any stock at a loss", "Buying stocks in a tax-advantaged account", "Claiming capital gains on short-term trades"], "p_correct_if_know": 0.68},
            {"id": "tax_filing_3_3", "question": "Roth IRA conversions are taxable in the year of conversion. The best year to convert is:", "answer": "A low-income year when you are in a lower tax bracket than expected in retirement", "distractors": ["Always the current year", "The year you turn 59.5", "The year markets are highest"], "p_correct_if_know": 0.65},
            {"id": "tax_filing_3_4", "question": "Bunching deductions is a strategy where you:", "answer": "Concentrate deductible expenses in alternate years to exceed the standard deduction", "distractors": ["Take the standard deduction every year", "Spread deductions evenly across years", "Deduct business expenses monthly"], "p_correct_if_know": 0.62},
            {"id": "tax_filing_3_5", "question": "Qualified Opportunity Zone investments allow you to:", "answer": "Defer and potentially reduce capital gains taxes by investing in designated low-income areas", "distractors": ["Avoid all federal taxes permanently", "Deduct investment losses twice", "Receive government matching on investments"], "p_correct_if_know": 0.55},
        ],
    },

    "index_funds": {
        1: [
            {"id": "index_funds_1_1", "question": "An index fund tracks:", "answer": "A market index like the S&P 500", "distractors": ["A fund manager's stock picks", "Government bond rates", "Real estate prices"], "p_correct_if_know": 0.93},
            {"id": "index_funds_1_2", "question": "Index funds typically have lower costs than actively managed funds because:", "answer": "They require minimal management - no analysts or active trading", "distractors": ["They are subsidized by the government", "They only hold stocks under $10", "They are only available through employers"], "p_correct_if_know": 0.90},
            {"id": "index_funds_1_3", "question": "An expense ratio of 0.03% vs 1.0% on a $100,000 portfolio means annual fees of:", "answer": "$30 vs $1,000", "distractors": ["$300 vs $100", "$3 vs $100", "$30 vs $100"], "p_correct_if_know": 0.88},
            {"id": "index_funds_1_4", "question": "The S&P 500 index tracks:", "answer": "500 of the largest publicly traded US companies", "distractors": ["All stocks on the NYSE", "The top 100 global companies", "All US bonds"], "p_correct_if_know": 0.87},
            {"id": "index_funds_1_5", "question": "Passive investing via index funds aims to:", "answer": "Match market returns rather than beat them", "distractors": ["Always outperform the market", "Avoid all market volatility", "Generate income through dividends only"], "p_correct_if_know": 0.89},
        ],
        2: [
            {"id": "index_funds_2_1", "question": "Research consistently shows most actively managed funds:", "answer": "Underperform their benchmark index over 10+ years", "distractors": ["Outperform index funds over long periods", "Match index funds exactly", "Are safer than index funds"], "p_correct_if_know": 0.82},
            {"id": "index_funds_2_2", "question": "A total market index fund holds:", "answer": "All publicly traded stocks in a market, weighted by market cap", "distractors": ["Only large-cap stocks", "Exactly 500 companies", "International stocks only"], "p_correct_if_know": 0.80},
            {"id": "index_funds_2_3", "question": "The difference in final portfolio value between 0.05% and 1% expense ratios over 30 years on $10,000?", "answer": "Approximately $30,000+ (compounding makes fees enormously costly)", "distractors": ["Approximately $2,850", "Approximately $1,000", "Negligible"], "p_correct_if_know": 0.78},
            {"id": "index_funds_2_4", "question": "A three-fund portfolio consists of:", "answer": "US total market + international + bonds - broad diversification at low cost", "distractors": ["S&P 500 + gold + real estate", "Growth + value + dividend funds", "Tech + healthcare + energy sector funds"], "p_correct_if_know": 0.75},
            {"id": "index_funds_2_5", "question": "A target-date fund automatically:", "answer": "Shifts from stocks to bonds as your retirement date approaches", "distractors": ["Picks the best-performing stocks each year", "Guarantees a return by your target date", "Stops investing once you retire"], "p_correct_if_know": 0.78},
        ],
        3: [
            {"id": "index_funds_3_1", "question": "Tax-loss harvesting with index funds involves:", "answer": "Selling a losing fund to realize a tax loss, then buying a similar (not identical) fund", "distractors": ["Selling winners to pay taxes then reinvesting", "Avoiding all taxable events permanently", "Holding funds only in tax-advantaged accounts"], "p_correct_if_know": 0.72},
            {"id": "index_funds_3_2", "question": "Why might a target-date fund be preferable to a DIY three-fund portfolio for many investors?", "answer": "Automatic rebalancing and glide path reduces behavioral risk from panic selling or inaction", "distractors": ["Higher returns historically", "Lower expense ratios always", "Greater tax efficiency"], "p_correct_if_know": 0.68},
            {"id": "index_funds_3_3", "question": "Smart beta (factor) ETFs differ from traditional index funds by:", "answer": "Weighting stocks by factors like value or momentum rather than pure market cap", "distractors": ["Tracking a different index each month", "Holding only bonds", "Charging no expense ratio"], "p_correct_if_know": 0.62},
            {"id": "index_funds_3_4", "question": "Asset location strategy means:", "answer": "Placing tax-inefficient assets (bonds, REITs) in tax-advantaged accounts and tax-efficient ones (index funds) in taxable accounts", "distractors": ["Buying real estate through index funds", "Locating the cheapest index funds", "Allocating based on geographic diversification"], "p_correct_if_know": 0.60},
            {"id": "index_funds_3_5", "question": "Tracking error in an index fund measures:", "answer": "How much the fund's returns deviate from its benchmark index", "distractors": ["The fund's total expense ratio", "Number of stocks not in the index", "Daily price volatility"], "p_correct_if_know": 0.58},
        ],
    },

    "retirement_planning": {
        1: [
            {"id": "retirement_1_1", "question": "The primary purpose of retirement planning is:", "answer": "Accumulating enough assets to fund living expenses without employment income", "distractors": ["Minimizing taxes during your working years", "Qualifying for Social Security", "Paying off your mortgage before age 65"], "p_correct_if_know": 0.92},
            {"id": "retirement_1_2", "question": "Social Security retirement benefits can first be claimed at age:", "answer": "62 (with reduced benefits)", "distractors": ["59.5", "65", "70"], "p_correct_if_know": 0.85},
            {"id": "retirement_1_3", "question": "Full Social Security retirement age for people born after 1960 is:", "answer": "67", "distractors": ["62", "65", "70"], "p_correct_if_know": 0.80},
            {"id": "retirement_1_4", "question": "Delaying Social Security from age 62 to 70 increases your monthly benefit by approximately:", "answer": "76% (8% per year from full retirement age to 70)", "distractors": ["10%", "25%", "50%"], "p_correct_if_know": 0.75},
            {"id": "retirement_1_5", "question": "The common rule of thumb for retirement savings by age 30 is:", "answer": "1x your annual salary", "distractors": ["3x your salary", "0.5x your salary", "No specific target at 30"], "p_correct_if_know": 0.78},
        ],
        2: [
            {"id": "retirement_2_1", "question": "The 4% rule suggests you can retire when your portfolio is worth how many times your annual expenses?", "answer": "25x (4% of 25x = 100% of annual expenses)", "distractors": ["10x", "15x", "50x"], "p_correct_if_know": 0.80},
            {"id": "retirement_2_2", "question": "Required Minimum Distributions (RMDs) from traditional IRAs start at age:", "answer": "73 (as of the SECURE 2.0 Act)", "distractors": ["59.5", "65", "70.5"], "p_correct_if_know": 0.72},
            {"id": "retirement_2_3", "question": "A pension (defined benefit plan) provides:", "answer": "A guaranteed monthly income in retirement based on years of service and salary", "distractors": ["A lump sum at retirement based on contributions", "Variable income depending on market performance", "The same benefit as a 401(k)"], "p_correct_if_know": 0.78},
            {"id": "retirement_2_4", "question": "The safe withdrawal rate accounts for:", "answer": "Inflation, sequence-of-returns risk, and a 30-year retirement horizon", "distractors": ["Only average market returns", "Tax rates in retirement", "Social Security income only"], "p_correct_if_know": 0.70},
            {"id": "retirement_2_5", "question": "Catch-up contributions to a 401(k) are available starting at age:", "answer": "50 (allowing an extra $7,500/year as of 2024)", "distractors": ["40", "59.5", "65"], "p_correct_if_know": 0.72},
        ],
        3: [
            {"id": "retirement_3_1", "question": "You need $60,000/year in retirement. Social Security pays $20,000. Portfolio needed using 4% rule?", "answer": "$1,000,000 (to generate $40,000/year at 4%)", "distractors": ["$1,500,000", "$600,000", "$500,000"], "p_correct_if_know": 0.72},
            {"id": "retirement_3_2", "question": "A Roth conversion ladder allows early retirees to:", "answer": "Access Roth IRA principal tax and penalty free before 59.5 after a 5-year holding period", "distractors": ["Withdraw from 401k penalty free at any age", "Convert unlimited funds tax free", "Avoid Required Minimum Distributions"], "p_correct_if_know": 0.62},
            {"id": "retirement_3_3", "question": "Sequence-of-returns risk is most dangerous in:", "answer": "The first 5-10 years of retirement when large withdrawals meet early losses", "distractors": ["The accumulation phase", "The last years of retirement", "Equally throughout retirement"], "p_correct_if_know": 0.65},
            {"id": "retirement_3_4", "question": "The bucket strategy in retirement involves:", "answer": "Dividing assets into short-term (cash), medium-term (bonds), long-term (stocks) buckets to manage withdrawals", "distractors": ["Investing in three index funds", "Withdrawing from accounts in alphabetical order", "Allocating 1/3 each to stocks, bonds, and real estate"], "p_correct_if_know": 0.60},
            {"id": "retirement_3_5", "question": "Medicare becomes available at age 65. Until then, retiring early means you must:", "answer": "Fund your own health insurance via COBRA, marketplace plans, or a spouse's plan", "distractors": ["Use Medicaid automatically", "Go without insurance by law", "Stay employed part-time to maintain coverage"], "p_correct_if_know": 0.65},
        ],
    },
}

# ---------------------------------------------------------------------------
# 3. SCENARIO BANK
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "scenario_001",
        "concepts_tested": ["budgeting", "emergency_fund"],
        "difficulty": 2,
        "prompt": "Maria earns $4,000/month take-home. Fixed expenses $2,200, variable expenses $800. No emergency fund. How should she allocate her surplus, and how long to build a 3-month emergency fund?",
        "correct_reasoning": "Surplus = $4,000 - $2,200 - $800 = $1,000/month. 3-month fund = 3 x $3,000 = $9,000. Timeline = 9 months.",
    },
    {
        "id": "scenario_002",
        "concepts_tested": ["compound_interest", "investment_basics", "index_funds"],
        "difficulty": 3,
        "prompt": "James, 25, invests $400/month. Option A: actively managed fund, 10% gross return, 1.2% expense ratio. Option B: index fund, 10% gross return, 0.04% expense ratio. Approximate difference at age 65?",
        "correct_reasoning": "Option A net: 8.8%. Option B net: 9.96%. Over 40 years: Option A ~$1.55M, Option B ~$2.35M. Difference ~$800,000.",
    },
    {
        "id": "scenario_003",
        "concepts_tested": ["credit_score", "apr", "amortization"],
        "difficulty": 3,
        "prompt": "Alex wants a $25,000 car loan for 5 years. 620 credit score: 9% APR. 740 score: 5% APR. How much does improving the score save?",
        "correct_reasoning": "At 9%: monthly ~$519, total interest ~$6,140. At 5%: monthly ~$472, total interest ~$3,307. Savings ~$2,833.",
    },
    {
        "id": "scenario_004",
        "concepts_tested": ["tax_advantaged_accounts", "investment_basics"],
        "difficulty": 2,
        "prompt": "Sarah earns $75,000, 22% bracket. Employer matches 100% of 401(k) up to 4% of salary. She contributes 2%. Annual cost of under-contribution?",
        "correct_reasoning": "Missing $1,500 in employer match. Plus 22% tax savings on additional $1,500 = $330. Total annual cost: ~$1,830.",
    },
    {
        "id": "scenario_005",
        "concepts_tested": ["amortization", "apr", "budgeting"],
        "difficulty": 3,
        "prompt": "Rachel has a $300,000 mortgage at 7%, 30 years (~$1,996/month). She can pay an extra $300/month toward principal. How does this change payoff timeline and total interest?",
        "correct_reasoning": "Extra $300/month reduces term by ~7-8 years. Interest savings ~$80,000-$100,000. Payoff in ~22-23 years.",
    },
    {
        "id": "scenario_006",
        "concepts_tested": ["emergency_fund", "insurance", "budgeting"],
        "difficulty": 2,
        "prompt": "Tom has a $1,000 car deductible and $2,500 health deductible. He has $800 in savings. Is he adequately prepared? What should he do first?",
        "correct_reasoning": "Tom cannot cover either deductible. First priority: build emergency fund to at least $2,500, then to 3 months of expenses.",
    },
    {
        "id": "scenario_007",
        "concepts_tested": ["tax_filing", "tax_advantaged_accounts"],
        "difficulty": 3,
        "prompt": "Nina earns $95,000. Contributes $10,000 to traditional 401(k). Has $4,000 mortgage interest and $6,000 charitable donations. Itemize or standard deduction? Taxable income?",
        "correct_reasoning": "Income after 401k: $85,000. Itemized: $10,000 < $14,600 standard. Take standard. Taxable income: ~$70,400.",
    },
    {
        "id": "scenario_008",
        "concepts_tested": ["simple_interest", "apr"],
        "difficulty": 2,
        "prompt": "A payday loan charges $15 per $100 borrowed for 2 weeks. What is the effective APR? How does this compare to a credit card at 24% APR?",
        "correct_reasoning": "APR = 15% x 26 two-week periods = 390%. Credit card at 24% APR is dramatically cheaper.",
    },
    {
        "id": "scenario_009",
        "concepts_tested": ["debt_payoff", "investment_basics"],
        "difficulty": 3,
        "prompt": "Jordan has $10,000 to allocate. $3,000 credit card at 22%, $7,000 student loan at 5%. Investment account averaging 8%. Emergency fund fully funded. Optimal allocation?",
        "correct_reasoning": "Pay off $3,000 credit card (22% > 8%). Invest remaining $7,000 (8% expected > 5% loan cost).",
    },
    {
        "id": "scenario_010",
        "concepts_tested": ["net_worth", "retirement_planning", "index_funds"],
        "difficulty": 3,
        "prompt": "Alex, 35, net worth $150,000 (401k $80k, home equity $50k, savings $20k). Earns $90,000/year, saves $15,000/year. On track to retire at 65 needing $60,000/year (Social Security covers $20,000)?",
        "correct_reasoning": "Need $40,000/year from portfolio. 4% rule: need $1,000,000. ~$100k investable growing at 7% + $15k/year contributions for 30 years ~$2.1M. Yes, on track.",
    },
    {
        "id": "scenario_011",
        "concepts_tested": ["simple_interest", "budgeting"],
        "difficulty": 1,
        "prompt": "Priya saves $200/month in an account paying 3% simple interest. How much will she have after 2 years from the savings plus interest on the cumulative balance?",
        "correct_reasoning": "Contributions: $200 x 24 = $4,800. Simple interest approximation on average balance ($2,400) at 3% for 2 years: ~$144. Total ~$4,944.",
    },
    {
        "id": "scenario_012",
        "concepts_tested": ["compound_interest", "emergency_fund"],
        "difficulty": 2,
        "prompt": "Devon needs a $12,000 emergency fund. He deposits $500/month into an HYSA paying 4.5% compounded monthly. About how long until he reaches the target?",
        "correct_reasoning": "Future value of annuity at 0.375%/month: $12,000 = 500 * [(1.00375^n - 1)/0.00375]. Solve: n ~= 23 months. Interest adds ~$500 over the period.",
    },
    {
        "id": "scenario_013",
        "concepts_tested": ["income_and_taxes", "budgeting"],
        "difficulty": 1,
        "prompt": "Lila earns $58,000 gross. Federal tax ~$5,200, FICA ~$4,440, state ~$2,300. What is her monthly take-home, and using 50/30/20 how much should go to needs?",
        "correct_reasoning": "Net annual: $58,000 - $11,940 = $46,060. Monthly: ~$3,838. Needs (50%) = ~$1,919.",
    },
    {
        "id": "scenario_014",
        "concepts_tested": ["credit_score", "debt_payoff"],
        "difficulty": 2,
        "prompt": "Marcus has three cards: $4,000 / $5,000 limit, $1,500 / $3,000 limit, $0 / $2,000 limit. Utilization is hurting his score. Which card should he pay down first to improve his score fastest, and what utilization should he target?",
        "correct_reasoning": "Total utilization: $5,500 / $10,000 = 55%. Card 1 (80%) is the worst individually. Pay it down first to bring utilization below 30% overall ($3,000 outstanding) and below 30% per card.",
    },
    {
        "id": "scenario_015",
        "concepts_tested": ["apr", "debt_payoff"],
        "difficulty": 1,
        "prompt": "Sam has two debts: $2,000 at 18% APR and $5,000 at 6% APR. He has $300/month extra after minimums. Using the avalanche method, which debt does he target first and why?",
        "correct_reasoning": "Avalanche targets highest APR: pay extra $300 on the 18% debt. Mathematically minimizes total interest because $1 paid on 18% debt saves more interest than $1 on 6% debt.",
    },
    {
        "id": "scenario_016",
        "concepts_tested": ["tax_advantaged_accounts", "retirement_planning"],
        "difficulty": 2,
        "prompt": "Yuki, 30, has $5,000/year to invest beyond her 401(k) match. She expects a higher tax bracket in retirement. Should she use a Roth IRA or a traditional IRA, and what is the approximate value at 65 at 7%?",
        "correct_reasoning": "Roth IRA preferred (pay 22% now, avoid 24%+ later). $5,000/year x 35 years at 7% future-value-of-annuity ~= $691,000 - all tax-free withdrawals.",
    },
    {
        "id": "scenario_017",
        "concepts_tested": ["insurance", "emergency_fund"],
        "difficulty": 1,
        "prompt": "Ben has a $1,500 health deductible and $800 in savings. He breaks his arm and faces a $1,500 bill. What does this say about his emergency fund, and what is his minimum next savings target?",
        "correct_reasoning": "He cannot cover the deductible. Minimum emergency fund should cover the largest deductible at minimum ($1,500), ideally 3-6 months of expenses. First savings target: $1,500.",
    },
    {
        "id": "scenario_018",
        "concepts_tested": ["amortization", "net_worth"],
        "difficulty": 2,
        "prompt": "Tara bought a $400,000 home with 20% down ($80,000) on a 30-year mortgage at 6%. After 5 years she has paid down approximately $24,000 of principal and the home has appreciated to $440,000. What is her home-equity contribution to net worth now?",
        "correct_reasoning": "Mortgage balance: $320,000 - $24,000 = $296,000. Home value: $440,000. Equity: $440,000 - $296,000 = $144,000.",
    },
    {
        "id": "scenario_019",
        "concepts_tested": ["index_funds", "investment_basics"],
        "difficulty": 1,
        "prompt": "Riya is choosing between an S&P 500 index fund (0.04% expense ratio) and an actively managed large-cap fund (0.95% expense ratio). Both target the same asset class. On a $20,000 investment held 20 years at 8% gross return, what is the approximate ending difference?",
        "correct_reasoning": "Index net 7.96%: ~$93,100. Active net 7.05%: ~$78,000. Difference ~$15,000. Fees compound dramatically.",
    },
    {
        "id": "scenario_020",
        "concepts_tested": ["tax_filing", "income_and_taxes"],
        "difficulty": 2,
        "prompt": "Aiden, single, earned $72,000 with $8,000 in 401(k) contributions. He has $5,000 mortgage interest, $4,000 state taxes, and $1,500 charitable. The standard deduction is $14,600. Itemize or standard? What is his taxable income?",
        "correct_reasoning": "Itemized total: $10,500 < $14,600 standard. Take standard. Taxable income: $72,000 - $8,000 - $14,600 = $49,400.",
    },
    {
        "id": "scenario_021",
        "concepts_tested": ["net_worth", "budgeting"],
        "difficulty": 1,
        "prompt": "Eliana lists: checking $2,000, savings $8,000, 401(k) $25,000, car $12,000, student loan $18,000, credit card $1,500. What is her net worth, and what one action this month most improves it given $400/month surplus?",
        "correct_reasoning": "Assets: $47,000. Liabilities: $19,500. Net worth: $27,500. Paying down the $1,500 credit card (likely highest APR) most improves net worth and reduces interest drag.",
    },
    {
        "id": "scenario_022",
        "concepts_tested": ["retirement_planning", "tax_advantaged_accounts", "compound_interest"],
        "difficulty": 3,
        "prompt": "Noah, 25, earns $70,000. Employer matches 100% of 401(k) up to 5%. He contributes 3%. If he raises to 5%, what is the approximate lifetime cost of his previous under-contribution by age 65 (7% return)?",
        "correct_reasoning": "Annual missed match: 2% x $70,000 = $1,400. Plus his missed 2% contribution: $1,400. Total missed annual savings: $2,800. FV at 7% over 40 years: ~$560,000.",
    },
    {
        "id": "scenario_023",
        "concepts_tested": ["emergency_fund", "debt_payoff"],
        "difficulty": 2,
        "prompt": "Sofia has $0 saved and $6,000 in credit card debt at 22% APR. She has $500/month surplus. What is the recommended split between building emergency fund and paying debt?",
        "correct_reasoning": "Standard guidance: build a starter emergency fund ($1,000-$1,500) first, then attack the 22% debt aggressively. After debt is gone, build to 3-6 months. Pure math says all $500 to debt, but a tiny buffer prevents re-borrowing on emergencies.",
    },
    {
        "id": "scenario_024",
        "concepts_tested": ["investment_basics", "insurance"],
        "difficulty": 2,
        "prompt": "Carlos, 38, married with two kids, earns $110,000. He has no life insurance but $80,000 in retirement accounts. Should he prioritize buying term life or increasing investment contributions this year?",
        "correct_reasoning": "Term life first. With dependents and inadequate assets to replace income, the downside of premature death is catastrophic. 20-year term covering 10x income (~$1M) is cheap (~$30-50/month at his age); buy it, then resume investing.",
    },
    {
        "id": "scenario_025",
        "concepts_tested": ["credit_score", "insurance"],
        "difficulty": 3,
        "prompt": "Marlene has a 590 credit score. Auto insurers in her state use credit-based insurance scores. She is quoted $2,400/year. If she raised her score to 720 over 18 months, insurers estimate ~25% lower premiums. What is the 5-year savings, and does this justify aggressive score improvement?",
        "correct_reasoning": "5-year savings: $2,400 x 0.25 x 5 = $3,000. Combined with loan APR improvements, credit-score lift typically yields several thousand dollars across products. Yes, justified.",
    },
    {
        "id": "scenario_026",
        "concepts_tested": ["apr", "amortization", "debt_payoff"],
        "difficulty": 3,
        "prompt": "Theo has a $250,000 mortgage at 6.5%, 30 years (~$1,580/month). He can pay $200/month extra to principal. Approximate years shaved off and interest saved over the life of the loan?",
        "correct_reasoning": "Extra $200/month: term reduced by ~6 years (to ~24 years), interest saved ~$70,000-$80,000.",
    },
    {
        "id": "scenario_027",
        "concepts_tested": ["tax_filing", "tax_advantaged_accounts", "retirement_planning"],
        "difficulty": 3,
        "prompt": "Vera, 45, earns $140,000 (24% bracket). She expects retirement income around $60,000/year (12% bracket). She has $7,000 to allocate. Traditional 401(k) deduction now vs Roth IRA conversion next year - which choice gives more after-tax retirement wealth, assuming 7% return for 20 years?",
        "correct_reasoning": "Traditional: $7,000 grows to ~$27,100, taxed at 12% on withdrawal = ~$23,850. Roth (already-taxed contribution of $7,000 - 24% tax = $5,320 net): grows to ~$20,600 tax-free. Traditional wins because bracket arbitrage (24%->12%) exceeds Roth's pure-growth advantage.",
    },
    {
        "id": "scenario_028",
        "concepts_tested": ["income_and_taxes", "compound_interest"],
        "difficulty": 2,
        "prompt": "Hassan earns $85,000. He gets a $5,000 raise pushing $1,500 into the 24% bracket (rest in 22%). If he invested the full after-tax raise at 7% for 25 years, what is the future value?",
        "correct_reasoning": "Tax on raise: $1,500 x 24% + $3,500 x 22% = $360 + $770 = $1,130. After-tax: $3,870. FV at 7% over 25 years: ~$21,000.",
    },
    {
        "id": "scenario_029",
        "concepts_tested": ["simple_interest", "apr", "credit_score"],
        "difficulty": 2,
        "prompt": "Devin needs $400 fast. Option A: payday loan, $60 fee for 2 weeks. Option B: credit card cash advance, 25% APR plus a 5% transaction fee. Which is cheaper if repaid in 2 weeks, and what is each option's effective APR?",
        "correct_reasoning": "Option A: $60 / $400 = 15% in 2 weeks -> APR ~390%. Option B: $20 fee + ~$3.85 interest (25%/26 periods) = ~$23.85. APR equivalent ~155%. Credit card is dramatically cheaper.",
    },
    {
        "id": "scenario_030",
        "concepts_tested": ["net_worth", "index_funds", "retirement_planning"],
        "difficulty": 3,
        "prompt": "Imani, 32, wants to retire at 50 (FIRE) needing $50,000/year. Current investable assets: $120,000. Annual savings: $40,000 into broad index funds. At 7% real return, can she reach her target in 18 years?",
        "correct_reasoning": "Target via 4% rule: $50,000 x 25 = $1.25M. FV: $120k x 1.07^18 + $40k x [(1.07^18 - 1) / 0.07] = ~$406k + ~$1.36M = ~$1.77M. Yes, comfortably on track.",
    },
    {
        "id": "scenario_031",
        "concepts_tested": ["budgeting", "debt_payoff", "emergency_fund"],
        "difficulty": 2,
        "prompt": "Camila has $300/month surplus, $0 emergency fund, $2,000 credit card debt at 21%, and a 6% student loan of $15,000. List the correct priority sequence and reasoning.",
        "correct_reasoning": "(1) Build $1,000 mini emergency fund (~3 months). (2) Pay credit card aggressively (21% beats any safe return). (3) Build EF to 3-6 months. (4) Pay 6% student loan above minimum. Avoids re-borrowing while attacking high-rate debt.",
    },
    {
        "id": "scenario_032",
        "concepts_tested": ["insurance", "retirement_planning"],
        "difficulty": 2,
        "prompt": "Reggie, 55, has $400,000 in retirement assets but no long-term care (LTC) insurance. Average LTC costs are $50,000-$100,000/year. Should he self-insure or buy LTC insurance, and what factors decide?",
        "correct_reasoning": "Self-insurance is risky at his asset level - a 3-year LTC stay could deplete most of his retirement. Hybrid LTC/life policies or stand-alone LTC purchased in mid-50s is typically cost-effective. Decision factors: asset cushion, family history, dependents.",
    },
    {
        "id": "scenario_033",
        "concepts_tested": ["income_and_taxes", "tax_advantaged_accounts"],
        "difficulty": 1,
        "prompt": "Anika earns $60,000 and contributes $6,000 to a traditional 401(k). If her marginal rate is 22%, how much does this save her in current-year federal income tax?",
        "correct_reasoning": "Tax savings: $6,000 x 22% = $1,320. Her effective contribution after tax savings is only $4,680 out of pocket.",
    },
    {
        "id": "scenario_034",
        "concepts_tested": ["compound_interest", "retirement_planning"],
        "difficulty": 2,
        "prompt": "Twins Mira and Ravi both want $1M by 65. Mira invests $300/month starting at 25. Ravi waits until 35 and invests $600/month. Both earn 7%. Who reaches $1M and how do their totals compare?",
        "correct_reasoning": "Mira (40 years, $300/mo): FV ~= $786,000. Ravi (30 years, $600/mo): FV ~= $735,000. Mira slightly ahead despite contributing half as much (~$144k vs $216k). Starting early dominates contribution rate.",
    },
    {
        "id": "scenario_035",
        "concepts_tested": ["credit_score", "amortization", "apr"],
        "difficulty": 2,
        "prompt": "Jia is buying a $350,000 home with 20% down. With a 660 score she's offered 7.5% APR. With a 760 score (achievable in 8 months) she's offered 6.5%. Monthly payment difference, and lifetime interest difference on the 30-year loan?",
        "correct_reasoning": "Loan: $280,000. At 7.5%: ~$1,958/mo, total interest ~$424,000. At 6.5%: ~$1,769/mo, total interest ~$357,000. Monthly savings ~$189; lifetime savings ~$67,000. Delaying to improve score is highly worthwhile.",
    },
    {
        "id": "scenario_036",
        "concepts_tested": ["investment_basics", "tax_filing"],
        "difficulty": 3,
        "prompt": "Dario realized $15,000 in long-term capital gains and $4,000 in short-term capital gains this year, plus $7,000 in long-term capital losses. He's in the 24% federal income bracket and 15% LTCG bracket. What is his net capital gains tax?",
        "correct_reasoning": "Losses first offset same-type gains: $7,000 LT loss offsets $7,000 of LT gain. Net LT gain: $8,000 at 15% = $1,200. ST gain $4,000 at 24% = $960. Total: $2,160.",
    },
    {
        "id": "scenario_037",
        "concepts_tested": ["budgeting", "simple_interest"],
        "difficulty": 1,
        "prompt": "Kai wants to save $3,000 for a vacation in 18 months. He puts a lump sum in a 4% simple-interest account today. How much must he deposit now to hit the goal, and what is the monthly savings equivalent if he instead saves nothing upfront?",
        "correct_reasoning": "Lump sum: $3,000 / (1 + 0.04 x 1.5) = ~$2,830. Monthly savings (ignoring interest): $3,000 / 18 = ~$167/month.",
    },
    {
        "id": "scenario_038",
        "concepts_tested": ["emergency_fund", "insurance", "budgeting"],
        "difficulty": 3,
        "prompt": "Leo, 28, has $25,000 saved, $4,000/month expenses, and is debating raising his auto insurance deductible from $500 to $2,000 to save $360/year in premiums. Given his emergency fund, is the deductible increase wise, and what is the break-even period?",
        "correct_reasoning": "His $25,000 EF easily absorbs the $1,500 deductible delta. Break-even: $1,500 / $360 ~= 4.2 accident-free years. Since most drivers go 5+ years without an at-fault claim, raising the deductible is statistically favorable.",
    },
    {
        "id": "scenario_039",
        "concepts_tested": ["retirement_planning", "investment_basics", "index_funds"],
        "difficulty": 3,
        "prompt": "Olufemi, 60, has $800,000 in a 95% stocks / 5% bonds portfolio. He retires in 5 years and needs $40,000/year (4% rule). Should he rebalance now, and to what allocation, to manage sequence-of-returns risk?",
        "correct_reasoning": "Yes. 95/5 is too aggressive entering the early-retirement risk zone. A glide toward 60/40 or 70/30 reduces sequence-of-returns risk. Rebalance gradually over the next 5 years to avoid timing risk while shifting to a sustainable retirement allocation.",
    },
    {
        "id": "scenario_040",
        "concepts_tested": ["debt_payoff", "credit_score", "budgeting"],
        "difficulty": 2,
        "prompt": "Esi has $18,000 in credit card debt across 4 cards (average APR 22%). She's offered a personal loan at 11% APR for 5 years to consolidate. Monthly cards minimums total $540; the consolidated loan payment is $391/month. What are the tradeoffs?",
        "correct_reasoning": "Pros: 11% APR halves interest cost vs 22%; single payment is simpler; freed cash flow ($149/month) accelerates payoff or builds EF. Cons: total interest paid depends on whether she stays disciplined (5-year term vs aggressive payoff); 0% cards become available again and tempt re-borrowing. Best path: consolidate AND throw the $149/month surplus back at the loan to shorten it.",
    },
]

# ---------------------------------------------------------------------------
# 4. CONSTANTS
# ---------------------------------------------------------------------------

NUM_CONCEPTS = len(CONCEPTS)
NUM_ACTIONS = len(ACTIONS)
MAX_EPISODE_STEPS = 50

BKT_DEFAULTS = {
    "p_learn": 0.30,
    "p_slip":  0.10,
    "p_guess": 0.20,
    "p_prior": 0.10,
}

# ---------------------------------------------------------------------------
# 5. BUILD AND SAVE
# ---------------------------------------------------------------------------

# Position within each (concept, difficulty) cell that is held out as the quiz
# item. With 5 items per cell, position 4 -> 4 practice + 1 quiz per cell.
QUIZ_POSITION = 4


def build_content(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "knowledge_graph.json", "w") as f:
        json.dump({"concepts": CONCEPTS, "graph": KNOWLEDGE_GRAPH, "num_concepts": NUM_CONCEPTS}, f, indent=2)
    print(f"  Wrote knowledge_graph.json  ({NUM_CONCEPTS} concepts)")

    items = []
    practice_items = []
    quiz_items = []
    for concept, by_diff in ITEM_BANK.items():
        for difficulty, qs in by_diff.items():
            for pos, q in enumerate(qs):
                set_label = "quiz" if pos == QUIZ_POSITION else "practice"
                item = {**q, "concept": concept, "difficulty": difficulty, "set": set_label}
                items.append(item)
                if set_label == "quiz":
                    quiz_items.append(item)
                else:
                    practice_items.append(item)

    item_bank_payload = {
        "actions": [{"index": i, "concept": c, "difficulty": d} for i, (c, d) in enumerate(ACTIONS)],
        "action_index": {f"{c}_{d}": i for i, (c, d) in enumerate(ACTIONS)},
        "items": items,
        "practice_items": practice_items,
        "quiz_items": quiz_items,
        "num_items": len(items),
        "num_practice": len(practice_items),
        "num_quiz": len(quiz_items),
        "num_actions": NUM_ACTIONS,
        "quiz_position": QUIZ_POSITION,
    }
    with open(output_dir / "item_bank.json", "w") as f:
        json.dump(item_bank_payload, f, indent=2)
    print(f"  Wrote item_bank.json        ({len(items)} items: {len(practice_items)} practice + {len(quiz_items)} quiz, {NUM_ACTIONS} actions)")

    with open(output_dir / "scenarios.json", "w") as f:
        json.dump({"scenarios": SCENARIOS, "num_scenarios": len(SCENARIOS)}, f, indent=2)
    print(f"  Wrote scenarios.json        ({len(SCENARIOS)} scenarios)")

    with open(output_dir / "constants.json", "w") as f:
        json.dump({
            "concepts": CONCEPTS,
            "difficulties": DIFFICULTIES,
            "num_concepts": NUM_CONCEPTS,
            "num_actions": NUM_ACTIONS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "bkt_defaults": BKT_DEFAULTS,
            "quiz_position": QUIZ_POSITION,
        }, f, indent=2)
    print(f"  Wrote constants.json")


def main():
    parser = argparse.ArgumentParser(description="Build finance tutoring curriculum content")
    parser.add_argument("--output_dir", type=str, default="content",
                        help="Directory to write content files (default: content/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Building content -> {output_dir}/")
    build_content(output_dir)

    total = sum(len(qs) for by_diff in ITEM_BANK.values() for qs in by_diff.values())
    n_cells = NUM_CONCEPTS * len(DIFFICULTIES)
    n_practice = n_cells * 4
    n_quiz = n_cells
    print(f"\nSanity check: {NUM_CONCEPTS} concepts x {len(DIFFICULTIES)} difficulties x 5 questions = {total} total questions")
    print(f"  -> {n_practice} practice + {n_quiz} quiz items (4 practice + 1 quiz per cell)")
    print(f"  -> {len(SCENARIOS)} multi-concept scenarios")
    assert total == 240, f"Expected 240 questions, got {total}"
    assert len(SCENARIOS) == 40, f"Expected 40 scenarios, got {len(SCENARIOS)}"
    print("Done.")


if __name__ == "__main__":
    main()