# Generated by Django 5.0.3 on 2024-03-07 15:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('zyraapp', '0003_alter_uploadedimage_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='uploadedimage',
            name='uploaded_at',
        ),
        migrations.AddField(
            model_name='uploadedimage',
            name='percentage',
            field=models.FloatField(null=True),
        ),
    ]